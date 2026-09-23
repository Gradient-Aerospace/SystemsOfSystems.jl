# A history is a record of a run. Only reasons with self-contained data are reconstructed;
# custom reasons and reasons holding live objects retain their identity and description.

function write_stop(group, stop; original_type = string(typeof(stop)), details = "")

    # Keep the description and diagnostics readable without decoding HDF5Vectors. The
    # stored value is always either a supported built-in reason or a self-contained record.
    group["type"] = original_type
    group["description"] = SystemsOfSystems.describe(stop)
    group["is_failure"] = stop isa SystemsOfSystems.AbstractFailureReason
    if !isempty(details)
        group["details"] = details
    end
    copy_to_hdf5_vector(group, "value", [stop]; chunk_length = 1)
    return nothing

end

# We exactly record stop reasons that we know how to reproduce. This is internal for now;
# custom reasons do not need to implement a persistence interface.
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

# Fallbacks preserve success/failure status without traversing live objects.
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

# Error diagnostics are text, so no process-specific stack-trace objects need to be loaded.
function record_stop(group, stop::SystemsOfSystems.EncounteredError)
    return record_stop(
        group,
        SystemsOfSystems.RecordedFailure(
            string(typeof(stop)), SystemsOfSystems.describe(stop),
            sprint(showerror, stop.exception, stop.trace),
        )
    )
end

function record_stop(
    group,
    stop::Union{SystemsOfSystems.RecordedStop, SystemsOfSystems.RecordedFailure}
)
    return write_stop(group, stop; stop.original_type, stop.details)
end

# Groups passed by callers remain open; their contents belong to the saved record.
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

    # Replacing a group must never delete another part of the run, including its source
    # log. Ancestor groups overlap just as much as identical groups do.
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

    clear_group(group)
    HDF5.attrs(group)["sim_history_version"] = 1
    group["log_path"] = log_path
    group["t_start"] = Float64(history.t_start)
    group["t_stop"] = Float64(history.t_stop)

    stop_group = HDF5.create_group(group, "stop")
    try
        record_stop(stop_group, history.stop)
    finally
        close(stop_group)
    end

    # Model persistence is explicitly requested. Unsupported models report an error.
    if save_model
        copy_to_hdf5_vector(group, "model", [history.model]; chunk_length = 1)
    end

    flush(HDF5.file(group))
    return nothing

end

function save_log_to_hdf5(group::HDF5.Group, log::AbstractLog; kwargs...)
    clear_group(group)
    save_mh_to_hdf5(group, log["/"]; kwargs...)
    return nothing
end

function save_log_to_hdf5(group::HDF5.Group, log::SystemsOfSystems.Logs.NullLog; kwargs...)
    clear_group(group)
    group["is_null"] = true
    return nothing
end

function save_log_to_hdf5(group::HDF5.Group, log::HDF5Log; kwargs...)

    # The source group is already the saved log when saving into its own location.
    if same_hdf5_file(group, log.group) && HDF5.name(group) == HDF5.name(log.group)
        return nothing
    end
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

check_history_destination(group, log::AbstractLog) = nothing
check_history_destination(group, log::HDF5Log) = check_separate_groups(group, log.group)

function save_sim_history(
    group::HDF5.Group,
    history::SystemsOfSystems.SimHistory;
    log_group::HDF5.Group,
    save_model = false,
)

    # A stored path identifies a group within this file, not an external dependency.
    if !same_hdf5_file(group, log_group)
        throw(ArgumentError("History and log groups must belong to the same HDF5 file."))
    end
    check_separate_groups(group, log_group)
    check_history_destination(group, history.log)
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

history_file(log::AbstractLog, filename) = nothing

function history_file(log::HDF5Log, filename)

    if !isvalid(log.group)
        error("Cannot save a history whose HDF5 log is closed.")
    end
    if !isfile(filename) || !samefile(filename, HDF5.filename(log.group))
        return nothing
    end
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

    # Reuse a live log's file and location. Other filenames describe new output files.
    fid = history_file(history.log, filename)
    if isnothing(log_path)
        log_path = isnothing(fid) ? "/log" : HDF5.name(history.log.group)
    end
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

    t_start = SystemsOfSystems.exact_time(read(group["t_start"]))
    t_stop = SystemsOfSystems.exact_time(read(group["t_stop"]))
    stop = load_hdf5_vector(group["stop/value"])[1]
    model = nothing
    if load_model && haskey(group, "model")
        model = load_hdf5_vector(group["model"])[1]
    end
    log_path = read(group["log_path"])
    log, _ = load_hdf5_log(HDF5.file(group), log_path)
    return SystemsOfSystems.SimHistory(t_start, t_stop, log, model, stop)

end

function load_sim_history(
    parent::Union{HDF5.File, HDF5.Group},
    path::AbstractString;
    kwargs...,
)

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

    # Filename loaders transfer ownership to the returned log. Group loaders borrow the
    # caller's file instead, so multiple histories can share an open results container.
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
    history = load_sim_history(filename; kwargs...)
    try
        return f(history)
    finally
        SystemsOfSystems.Logs.close_log(history.log)
    end
end

function own_log_file(log::HDF5Log, fid)
    log.fid = fid
    return nothing
end

function own_log_file(log::SystemsOfSystems.Logs.NullLog, fid)
    close(fid)
    return nothing
end
