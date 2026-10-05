# History persistence combines a log group with a separate group of run metadata. The
# metadata records the log's location within the file, allowing the two groups to be placed
# wherever the caller needs them. An existing HDF5 log can stay in place while its history
# metadata is saved alongside it.
#
# Saved termination reasons retain their API results without restoring hooks, tasks,
# or exception objects.

using SystemsOfSystems: TerminationSummary, CleanupErrorSummary

# Saves the termination reason to a group.
function write_stop(group, stop)
    summary = TerminationSummary(stop)
    for f in fieldnames(TerminationSummary)
        group[string(f)] = getproperty(summary, f)
    end
    return nothing
end

function read_stop(group)
    return TerminationSummary(;
        type = read(group["type"]),
        finished = read(group["finished"]),
        failed = read(group["failed"]),
        interrupted = read(group["interrupted"]),
        summary = read(group["summary"]),
        details = read(group["details"]),
    )
end

function save_cleanup_errors(group, errors)
    group["count"] = length(errors)
    for (index, err) in enumerate(errors)
        entry = HDF5.create_group(group, string(index))
        try
            summary = CleanupErrorSummary(err)
            for field in fieldnames(CleanupErrorSummary)
                entry[string(field)] = getproperty(summary, field)
            end
        finally
            close(entry)
        end
    end
    return nothing
end

function load_cleanup_errors(group)
    errors = SystemsOfSystems.AbstractCleanupError[]
    for index in 1:read(group["count"])
        entry = group[string(index)]
        try
            push!(errors, CleanupErrorSummary(;
                type = haskey(entry, "type") ? read(entry["type"]) : "",
                context = read(entry["context"]),
                details = read(entry["details"]),
            ))
        finally
            close(entry)
        end
    end
    return errors
end

# Saving replaces the contents of a destination group, including optional fields from an
# earlier save. Keep the group itself open so caller-owned handles remain usable.
function clear_group(group)
    for name in keys(group)
        HDF5.delete_object(group, name)
    end
    for name in keys(HDF5.attributes(group))
        HDF5.delete_attribute(group, name)
    end
    return nothing
end

function storage_group(parent, path)
    return haskey(parent, path) ? parent[path] : HDF5.create_group(parent, path)
end

# HDF5's file number identifies an open file independently of its filename, aliases,
# or the working directory. References identify objects within that file, not their paths.
function same_hdf5_file(a, b)
    return HDF5.API.h5o_get_info(a).fileno == HDF5.API.h5o_get_info(b).fileno
end

function same_hdf5_group(a, b)
    return same_hdf5_file(a, b) && HDF5.Reference(a, ".") == HDF5.Reference(b, ".")
end

function contains_hdf5_group(parent, child)

    # Compare the actual objects along the child's path. A hard or soft link may give
    # an ancestor another name, so comparing path prefixes alone can miss an overlap.
    reference = HDF5.Reference(parent, ".")
    names = split(HDF5.name(child), '/'; keepempty = false)
    for n in 0:length(names)
        path = "/" * join(names[1:n], "/")
        if HDF5.Reference(child, path) == reference
            return true
        end
    end
    return false

end

function check_separate_groups(a, b)

    # Clearing either destination must not remove the other group or a source log.
    # Reference comparisons are meaningful only within the same file.
    if same_hdf5_file(a, b) && (contains_hdf5_group(a, b) || contains_hdf5_group(b, a))
        throw(ArgumentError("History and log storage groups must not overlap."))
    end
    return nothing

end

function save_history_metadata(group, history, log_path; save_model)

    # Replace the previous run's metadata as a whole so omitted optional fields cannot
    # survive a later save. The log remains in its own group, identified by log_path.
    # Floating-point times match the representation used for time-series samples.
    clear_group(group)
    write_format_version(group, "sim_history_version", history_format_version)
    group["log_path"] = log_path
    group["t_start"] = Float64(history.t_start)
    group["t_stop"] = Float64(history.t_stop)

    # Save the termination API results as ordinary datasets. Close the temporary group
    # handle even if writing fails: the caller may keep the file open.
    stop_group = HDF5.create_group(group, "stop")
    try
        write_stop(stop_group, history.stop)
    finally
        close(stop_group)
    end

    # Cleanup failures are readable text because live exceptions and traces cannot be
    # reconstructed from a saved run. Older histories have no cleanup_errors group.
    cleanup_group = HDF5.create_group(group, "cleanup_errors")
    try
        save_cleanup_errors(cleanup_group, history.cleanup_errors)
    finally
        close(cleanup_group)
    end

    # Models can hold resources that are unsuitable for persistence, so saving one is
    # opt-in. A requested model that HDF5Vectors cannot encode must fail the save rather
    # than silently disappear from the record.
    if save_model
        copy_to_hdf5_vector(group, "model", [history.model]; chunk_length = 1)
    end

    # The file may remain open for further work; flush the completed record before return.
    flush(HDF5.file(group))
    return nothing

end

# In-memory logs are written from the root ModelHistory, which contains the whole model
# tree. Replacing the destination also removes variables omitted from the new log.
function save_log_to_hdf5(group::HDF5.Group, log::AbstractLog; kwargs...)
    clear_group(group)
    write_format_version(group, "log_format_version", log_format_version)
    save_mh_to_hdf5(group, log["/"]; kwargs...)
    return nothing
end

# A disabled log has no ModelHistory tree. Its marker distinguishes intentional absence
# of logging from a log group whose expected data is missing.
function save_log_to_hdf5(group::HDF5.Group, log::SystemsOfSystems.Logs.NullLog; kwargs...)
    clear_group(group)
    write_format_version(group, "log_format_version", log_format_version)
    group["is_null"] = true
    return nothing
end

function save_log_to_hdf5(group::HDF5.Group, log::HDF5Log; kwargs...)

    # Copying or reusing a log does not convert its representation. Preserve its version
    # and writer provenance, including their absence on legacy logs, and reject explicit
    # unsupported versions before modifying a destination.
    check_format_version(log.group, "log_format_version", log_format_version;
        allow_unversioned = true)

    # An HDF5 log already contains the saved samples and metadata. When its destination
    # is the same group, leave it intact so the simulation's open dataset handles continue
    # to refer to the original data.
    if same_hdf5_group(group, log.group)
        return nothing
    end

    # For a different destination, copy through HDF5 instead of reading every sample into
    # Julia. Check for overlap before clearing the destination, and preserve group
    # attributes as well as child objects so additional log metadata travels with the data.
    check_separate_groups(group, log.group)
    clear_group(group)
    for name in keys(log.group)
        HDF5.copy_object(log.group, name, group, name)
    end
    for name in keys(HDF5.attributes(log.group))
        HDF5.attributes(group)[name] = read(HDF5.attributes(log.group)[name])
    end
    return nothing

end

# In-memory logs cannot be erased by replacing an HDF5 group. A disk-backed source log
# needs an additional check even when it is being copied to a different destination group.
check_history_destination(group, log::AbstractLog) = nothing
check_history_destination(group, log::HDF5Log) = check_separate_groups(group, log.group)

function save_sim_history(
    group::HDF5.Group,
    history::SystemsOfSystems.SimHistory;
    log_group::HDF5.Group,
    save_model = false,
)

    # The metadata locates its log by an absolute path within the same file. Validate the
    # destinations before replacing anything, including a source log that might otherwise
    # be removed when the history metadata is cleared.
    if !same_hdf5_file(group, log_group)
        throw(ArgumentError("History and log groups must belong to the same HDF5 file."))
    end
    check_separate_groups(group, log_group)
    check_history_destination(group, history.log)

    # Write or reuse the log first, then record its location with the run metadata. The
    # group methods leave the caller responsible for the lifetime of the containing file.
    save_log_to_hdf5(log_group, history.log)
    save_history_metadata(group, history, HDF5.name(log_group); save_model)
    return nothing

end

function save_sim_history(
    parent::Union{HDF5.File, HDF5.Group},
    path::AbstractString,
    history::SystemsOfSystems.SimHistory;
    log_path = "log",
    save_model = false,
)

    # Path-based saving opens its own group handles. Close those handles after the write,
    # including on failure, without closing the parent supplied by the caller.
    group = storage_group(parent, path)
    try
        log_group = storage_group(parent, log_path)
        try
            save_sim_history(group, history; log_group, save_model)
        finally
            close(log_group)
        end
    finally
        close(group)
    end
    return nothing

end

# Return the log's existing file only when it is also the requested destination. Otherwise
# the filename-based saver opens a new output file. In-memory logs have no file to reuse.
history_file(log::AbstractLog, filename) = nothing

function history_file(log::HDF5Log, filename)

    # Inspect an existing HDF5 destination without truncating it. Comparing open-file
    # identities also works when the source was opened with a relative name and the
    # caller has since changed directories or renamed the file.
    if !isvalid(log.group)
        error("Cannot save a history whose HDF5 log is closed.")
    end
    if !HDF5.ishdf5(filename)
        return nothing
    end

    # HDF5 requires repeated opens of a file to agree on its file-close behavior. Reuse
    # the source's access properties, including when its handle belongs to the caller.
    properties = HDF5.get_access_properties(HDF5.file(log.group))
    same_file = try

        HDF5.h5open(filename, "r"; fapl = properties) do destination
            same_hdf5_file(destination, log.group)
        end

    finally

        close(properties)

    end
    if !same_file
        return nothing
    end

    # Reusing a source file writes metadata into it. A read-only source is still suitable
    # for copying elsewhere, but cannot be used as an in-place destination.
    fid = HDF5.file(log.group)
    if HDF5.API.h5f_get_intent(fid) == HDF5.API.H5F_ACC_RDONLY
        error("Saving to the log's existing file requires a writable HDF5 log.")
    end
    return fid

end

function save_sim_history(
    filename::AbstractString,
    history::SystemsOfSystems.SimHistory;
    history_path = "/history",
    log_path = nothing,
    save_model = false,
)

    # By default, keep a live log at its existing location when saving into its own file.
    # New output files use /log; an explicit log_path overrides either default.
    fid = history_file(history.log, filename)
    if isnothing(log_path)
        log_path = isnothing(fid) ? "/log" : HDF5.name(history.log.group)
    end

    # Only files opened here are closed here. A reused file belongs to the existing log
    # and must remain open for the caller's continued access to its time series.
    if isnothing(fid)
        HDF5.h5open(filename, "w") do output
            save_sim_history(output, history_path, history; log_path, save_model)
        end
    else
        save_sim_history(fid, history_path, history; log_path, save_model)
    end
    return nothing

end

function load_sim_history(group::HDF5.Group; load_model = false)

    # Validate the format before reading fields or deserializing values. Writer package
    # versions are provenance only; they do not decide whether this layout is supported.
    check_format_version(group, "sim_history_version", history_format_version)

    # Restore the small run record eagerly. SimHistory uses exact simulation times even
    # though the file stores them as floats. Restore the termination reason as a summary.
    t_start = SystemsOfSystems.exact_time(read(group["t_start"]))
    t_stop = SystemsOfSystems.exact_time(read(group["t_stop"]))
    stop = read_stop(group["stop"])
    cleanup_errors = haskey(group, "cleanup_errors") ?
        load_cleanup_errors(group["cleanup_errors"]) :
        SystemsOfSystems.AbstractCleanupError[]

    # Leave the model entry unread unless requested. This lets callers inspect results
    # without requiring the saved model's types and resources to be reconstructible.
    model = nothing
    if load_model && haskey(group, "model")
        model = load_hdf5_vector(group["model"])[1]
    end

    # Follow the saved location instead of assuming /log. Loading the log builds the model
    # history tree but leaves samples backed by HDF5, so the caller's file must stay open.
    log_path = read(group["log_path"])
    log, _ = load_hdf5_log(HDF5.file(group), log_path)
    return SystemsOfSystems.SimHistory(t_start, t_stop, log, model, stop, cleanup_errors)

end

function load_sim_history(
    parent::Union{HDF5.File, HDF5.Group},
    path::AbstractString;
    kwargs...,
)

    # Unlike parent[path] passed directly to the group method, this convenience method
    # releases its temporary metadata handle immediately. The log has separate handles
    # for the data it retains, and the parent file remains the caller's responsibility.
    group = parent[path]
    try
        return load_sim_history(group; kwargs...)
    finally
        close(group)
    end

end

function load_sim_history(
    filename::AbstractString;
    history_path = "/history",
    load_model = false,
)

    # This overload opens the file, so it must either transfer responsibility for closing
    # it to the returned log or close it on failure. A NullLog retains no datasets and lets
    # own_log_file close the file immediately. Group-based callers keep file ownership.
    fid = HDF5.h5open(filename, "r")
    try
        history = load_sim_history(fid, history_path; load_model)
        own_log_file(history.log, fid)
        return history
    catch
        close(fid)
        rethrow()
    end

end

function load_sim_history(f::Function, filename::AbstractString; kwargs...)

    # Keep the file open only for the callback. Its result is returned unchanged, so the
    # callback must copy any samples it needs afterward instead of returning HDF5 handles.
    history = load_sim_history(filename; kwargs...)
    try
        return f(history)
    finally
        SystemsOfSystems.Logs.close_log(history.log)
    end

end

# Group loaders initially borrow their file. Filename loaders use these methods to give
# a disk-backed log ownership, or to close a file that a NullLog no longer needs.
function own_log_file(log::HDF5Log, fid)
    log.fid = fid
    return nothing
end

function own_log_file(log::SystemsOfSystems.Logs.NullLog, fid)
    close(fid)
    return nothing
end
