# History persistence combines a log group with a separate group of run metadata. The
# metadata records the log's location within the file, allowing the two groups to be placed
# wherever the caller needs them. An existing HDF5 log can stay in place while its history
# metadata is saved alongside it.
#
# Saved histories describe completed runs. Termination reasons that contain hooks, tasks,
# or exceptions become descriptive records rather than attempts to restore those objects.

function write_stop(group, stop; original_type = string(typeof(stop)), details = "")

    # Ordinary datasets let HDF5 readers inspect the termination without understanding
    # Julia types. For a descriptive record, original_type identifies the reason from the
    # run rather than the RecordedStop or RecordedFailure used to store it.
    group["type"] = original_type
    group["description"] = SystemsOfSystems.describe(stop)
    group["is_failure"] = stop isa SystemsOfSystems.AbstractFailureReason
    if !isempty(details)
        group["details"] = details
    end

    # The Julia loader also needs a value it can reconstruct. record_stop supplies either
    # a supported built-in reason or a descriptive record; a one-element HDF5Vector lets
    # both use the same encoding as other structured values in the file.
    copy_to_hdf5_vector(group, "value", [stop]; chunk_length = 1)
    return nothing

end

# These built-in reasons contain only data that HDF5Vectors can restore. Keeping their
# concrete types preserves useful fields, such as the stopping time or model path.
function record_stop(
    group,
    stop::Union{
        SystemsOfSystems.UnknownStopReason,
        SystemsOfSystems.ReachedEndTime,
        SystemsOfSystems.ModelRequestedStop,
        SystemsOfSystems.Interrupted,
        SystemsOfSystems.Solvers.SolverFailedToConverge,
        SystemsOfSystems.Solvers.SolverStepSizeUnderflow,
    }
)
    return write_stop(group, stop)
end

# Other reasons may contain resources that cannot be restored outside the original run.
# Their type name and description still explain the termination. A separate failure method
# preserves the classification used by succeeded(history), without inspecting those fields.
function record_stop(group, stop::SystemsOfSystems.AbstractTerminationReason)
    return record_stop(
        group,
        SystemsOfSystems.RecordedStop(
            string(typeof(stop)), SystemsOfSystems.describe(stop),
        )
    )
end

function record_stop(group, stop::SystemsOfSystems.AbstractFailureReason)
    return record_stop(
        group,
        SystemsOfSystems.RecordedFailure(
            string(typeof(stop)), SystemsOfSystems.describe(stop),
        )
    )
end

# An exception's description alone omits the information needed to diagnose it. Rendering
# the exception and stack trace preserves those diagnostics without saving exception
# payloads or compiler objects that may be meaningful only in the originating process.
function record_stop(group, stop::SystemsOfSystems.EncounteredError)
    return record_stop(
        group,
        SystemsOfSystems.RecordedFailure(
            string(typeof(stop)), SystemsOfSystems.describe(stop),
            sprint(showerror, stop.exception, stop.trace),
        )
    )
end

# A loaded history may be saved again. Keep the original reason's identity and diagnostics
# rather than replacing them with the type name of its descriptive record.
function record_stop(
    group,
    stop::Union{SystemsOfSystems.RecordedStop, SystemsOfSystems.RecordedFailure}
)
    return write_stop(group, stop; stop.original_type, stop.details)
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

function same_hdf5_file(a, b)
    return samefile(HDF5.filename(a), HDF5.filename(b))
end

function check_separate_groups(a, b)

    # Clearing a destination would also remove any log or history stored beneath it.
    # Reject equal paths and ancestor/descendant paths within the same file before writing.
    # Trailing separators keep sibling names such as /run and /run2 from matching.
    if same_hdf5_file(a, b)
        ap = rstrip(HDF5.name(a), '/') * "/"
        bp = rstrip(HDF5.name(b), '/') * "/"
        if startswith(ap, bp) || startswith(bp, ap)
            throw(ArgumentError("History and log storage groups must not overlap."))
        end
    end
    return nothing

end

function save_history_metadata(group, history, log_path; save_model)

    # Replace the previous run's metadata as a whole so omitted optional fields cannot
    # survive a later save. The log remains in its own group, identified by log_path.
    # Floating-point times match the representation used for time-series samples.
    clear_group(group)
    HDF5.attrs(group)["sim_history_version"] = 1
    group["log_path"] = log_path
    group["t_start"] = Float64(history.t_start)
    group["t_stop"] = Float64(history.t_stop)

    # The termination writer chooses a restorable representation for the reason. Release
    # its temporary group handle even if writing fails: the caller may keep the file open.
    stop_group = HDF5.create_group(group, "stop")
    try
        record_stop(stop_group, history.stop)
    finally
        close(stop_group)
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
    save_mh_to_hdf5(group, log["/"]; kwargs...)
    return nothing
end

# A disabled log has no ModelHistory tree. Its marker distinguishes intentional absence
# of logging from a log group whose expected data is missing.
function save_log_to_hdf5(group::HDF5.Group, log::SystemsOfSystems.Logs.NullLog; kwargs...)
    clear_group(group)
    group["is_null"] = true
    return nothing
end

function save_log_to_hdf5(group::HDF5.Group, log::HDF5Log; kwargs...)

    # An HDF5 log already contains the saved samples and metadata. When its destination
    # is the same group, leave it intact so the simulation's open dataset handles continue
    # to refer to the original data.
    if same_hdf5_file(group, log.group) && HDF5.name(group) == HDF5.name(log.group)
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
    log_path = "/log",
    save_model = false,
)

    # Path-based saving opens its own group handles. Close those handles after the write,
    # including on failure, without closing the parent supplied by the caller.
    group = storage_group(parent, path)
    log_group = storage_group(parent, log_path)
    try
        save_sim_history(group, history; log_group, save_model)
    finally
        close(group)
        close(log_group)
    end
    return nothing

end

# Return the log's existing file only when it is also the requested destination. Otherwise
# the filename-based saver opens a new output file. In-memory logs have no file to reuse.
history_file(log::AbstractLog, filename) = nothing

function history_file(log::HDF5Log, filename)

    # The source must remain readable even when saving to another file. Compare actual
    # files rather than filename strings so relative paths and aliases do not cause an
    # open source file to be mistaken for a new destination and truncated.
    if !isvalid(log.group)
        error("Cannot save a history whose HDF5 log is closed.")
    end
    if !isfile(filename) || !samefile(filename, HDF5.filename(log.group))
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

    # Restore the small run record eagerly. SimHistory uses exact simulation times even
    # though the file stores them as floats; the saved stop value is a built-in reason or
    # one of the descriptive records selected by record_stop.
    t_start = SystemsOfSystems.exact_time(read(group["t_start"]))
    t_stop = SystemsOfSystems.exact_time(read(group["t_stop"]))
    stop = load_hdf5_vector(group["stop/value"])[1]

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
    return SystemsOfSystems.SimHistory(t_start, t_stop, log, model, stop)

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
