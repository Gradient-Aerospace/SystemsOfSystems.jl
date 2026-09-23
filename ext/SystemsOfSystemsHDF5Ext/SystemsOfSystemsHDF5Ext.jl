module SystemsOfSystemsHDF5Ext

using OrderedCollections: OrderedDict
import HDF5
import Serialization
import SystemsOfSystems
import SystemsOfSystems: save_sim_history, load_sim_history
using HDF5Vectors: create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector

using SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription
using SystemsOfSystems.LoggingPolicies: is_variable_in_set
using SystemsOfSystems.Logs: ModelHistory, AbstractLogOptions, AbstractLog,
    HDF5LogOptions, create_time_series_for_model!

import SystemsOfSystems.Logs: create_log, create_time_series_for_var,
    record_model_description, close_log,
    load_hdf5_log, save_log_to_hdf5,
    save_time_series_to_hdf5, load_time_series_from_hdf5

"""
    HDF5Log(fid, group, model_history_dict)

A container for the same continuous and discrete states, outputs, and metadata as a
`BasicLog`, with an HDF5 file as the underlying storage. Constants that cannot be
represented by HDF5Vectors are omitted with a warning. This prevents the complete log from
being stored in RAM, which is critical for very long simulations. However, it is much
slower than a `BasicLog`.

If you're just looking to have an HDF5 file artifact, it's faster to use a BasicLog and then
use `save_log_to_hdf5` when the simulation is over.

The `group` field identifies the log's storage group, independently of whether it is at
the default `/log` group or another group selected by the caller.

See [`HDF5LogOptions`](@ref) for more.
"""
mutable struct HDF5Log <: AbstractLog
    fid::Union{HDF5.File, Nothing}
    group::HDF5.Group
    model_history_dict::OrderedDict{String, ModelHistory}
end

Base.setindex!(log::HDF5Log, mh, slug) = (log.model_history_dict[slug] = mh)
Base.getindex(log::HDF5Log, k) = log.model_history_dict[k]
Base.keys(log::HDF5Log) = keys(log.model_history_dict)
Base.values(log::HDF5Log) = values(log.model_history_dict)
Base.pairs(log::HDF5Log) = pairs(log.model_history_dict)

# We won't log missings, so extract the "real" type to log from unions with missings.
figure_out_el_type(::Type{Union{Missing, T}}) where {T} = T
figure_out_el_type(::Type{T}) where {T} = T

function serialize_to_bytes(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

function deserialize_from_bytes(bytes)
    return Serialization.deserialize(IOBuffer(bytes))
end

function record_dimensions(group, dimensions)
    group["labels"] = [dimension.label for dimension in dimensions]
    group["units"] = [dimension.units for dimension in dimensions]
    return nothing
end

function record_groups(group, groups)

    group["groups_are_missing"] = ismissing(groups)
    groups_to_record = ismissing(groups) ? Pair{String, Vector{String}}[] : groups
    group["group_labels"] = String[label for (label, _) in groups_to_record]
    group["group_dimension_counts"] = Int[
        length(labels) for (_, labels) in groups_to_record
    ]
    group["group_dimension_labels"] = String[
        dimension_label
        for (_, dimension_labels) in groups_to_record
        for dimension_label in dimension_labels
    ]
    return nothing

end

function record_interpolator(group, interpolator)

    group["interpolator_type"] = string(typeof(interpolator))
    group["serialized_interpolator"] = serialize_to_bytes(interpolator)
    return nothing

end

function record_time_series_metadata(group, ts)

    group["title"] = ts.title
    group["path"] = ts.path
    group["discrete"] = ts.discrete
    group["time_label"] = ts.time_dimension.label
    group["time_units"] = ts.time_dimension.units
    record_dimensions(group, ts.dimensions)
    record_groups(group, ts.groups)
    record_interpolator(group, ts.interpolator)
    return nothing

end

function create_time_series_for_var(
    log::HDF5Log,
    breadcrumbs,
    var_name,
    var::VariableDescription{T},
    time_dimension;
    discrete,
) where {T}

    el_type = figure_out_el_type(T)
    group_path = join(
        (
            ("models/$el" for el in breadcrumbs)...,
            "timeseries",
            var_name,
        ),
        "/",
    )
    slug = join("/" * model for model in breadcrumbs) * "/" * var_name
    group = HDF5.create_group(log.group, group_path)
    ts = TimeSeries(;
        var.title,
        time = create_hdf5_vector(group, "time", Float64),
        data = create_hdf5_vector(group, "data", el_type),
        time_dimension,
        var.dimensions,
        path = slug,
        discrete,
        var.interpolator,
        var.groups,
    )
    record_time_series_metadata(group, ts)
    return ts

end

function create_time_series_for_var(
    log::HDF5Log,
    breadcrumbs,
    var_name,
    var::T,
    time_dimension;
    discrete,
) where {T}

    el_type = figure_out_el_type(T)
    group_path = join(
        (
            ("models/$el" for el in breadcrumbs)...,
            "timeseries",
            var_name,
        ),
        "/",
    )
    slug = join("/" * model for model in breadcrumbs) * "/" * var_name
    group = HDF5.create_group(log.group, group_path)
    ts = TimeSeries(;
        title = slug,
        time = create_hdf5_vector(group, "time", Float64),
        data = create_hdf5_vector(group, "data", el_type),
        time_dimension,
        path = slug,
        discrete,
    )
    record_time_series_metadata(group, ts)
    return ts

end

function record_constant(
    constant_group,
    v::VariableDescription{T},
    breadcrumbs,
    name,
) where {T}

    constant_group["is_variable_description"] = true
    constant_group["value_type"] = string(T)
    constant_group["serialized_value_type"] = serialize_to_bytes(T)
    constant_group["value_is_missing"] = ismissing(v.value)
    constant_group["title"] = v.title
    type = figure_out_el_type(T)
    vec = create_hdf5_vector(constant_group, "value", type; chunk_length = 1)
    if !ismissing(v.value)
        push!(vec, v.value)
    end
    record_dimensions(constant_group, v.dimensions)
    record_groups(constant_group, v.groups)
    record_interpolator(constant_group, v.interpolator)
    return nothing

end

function record_constant(constant_group, v, breadcrumbs, name)

    constant_group["is_variable_description"] = false
    constant_group["value_is_missing"] = ismissing(v)
    constant_group["title"] = join("/" * el for el in breadcrumbs) * "/$name"
    vec = create_hdf5_vector(constant_group, "value", typeof(v); chunk_length = 1)
    if !ismissing(v)
        push!(vec, v)
    end
    record_dimensions(constant_group, Dimension[])
    return nothing

end

function record_model_description(
    log::HDF5Log,
    breadcrumbs,
    md::ModelDescription,
    variable_set,
)

    # Get the ground started.
    group_path = join(("models/$el" for el in breadcrumbs), "/")
    if !isempty(group_path)
        group = HDF5.create_group(log.group, group_path)
    else
        group = log.group # This exists at creation.
    end

    # Keep a readable name for general HDF5 inspection and the actual Julia type for an
    # exact round trip when its defining module is available.
    group["type"] = string(md.type)
    group["serialized_type"] = serialize_to_bytes(md.type)

    # We save the constants and use a try-catch, because we don't know how to save
    # everything a user might have as a constant to HDF5, and we'd rather just omit
    # constants than throw an error. We'll give a warning though.
    constants_group = HDF5.create_group(group, "constants")
    saved_constants = String[]
    for (k, v) in pairs(md.constants)
        if !is_variable_in_set(k, variable_set)
            continue
        end
        constant_group = HDF5.create_group(constants_group, string(k))
        try
            record_constant(constant_group, v, breadcrumbs, k)
            push!(saved_constants, string(k))
        catch err
            trace = catch_backtrace()
            p = join("/" * el for el in breadcrumbs) * "/$k"
            message = "Failed to record the $p constant of type $(typeof(v)) in the " *
                "HDF5 output file. Skipping."
            @warn message exception = (err, trace)
            HDF5.delete_object(constant_group)
        end
    end

    # We record the names of each type of thing. This helps us know if, say, "position" is a
    # state or output or constant.
    names_group = HDF5.create_group(group, "names")
    names_group["constants"] = saved_constants
    names_group["continuous_states"] = String[
        string(k)
        for k in keys(md.continuous_states) if is_variable_in_set(k, variable_set)
    ]
    names_group["discrete_states"] = String[
        string(k)
        for k in keys(md.discrete_states) if is_variable_in_set(k, variable_set)
    ]
    names_group["continuous_outputs"] = String[
        string(k)
        for k in keys(md.continuous_outputs) if is_variable_in_set(k, variable_set)
    ]
    names_group["discrete_outputs"] = String[
        string(k)
        for k in keys(md.discrete_outputs) if is_variable_in_set(k, variable_set)
    ]
    names_group["models"] = String[string(k) for k in keys(md.models)]

    return nothing

end

function create_log(options::HDF5LogOptions, model_description, time_dimension)
    mkpath(dirname(options.filename))
    fid = HDF5.h5open(options.filename, "w")
    mhd = OrderedDict{String, ModelHistory}()
    log = HDF5Log(fid, storage_group(fid, options.path), mhd)
    logging_policy = options.logging_policy
    finalizer(close_log, log) # Close the file when this goes out of scope.
    breadcrumbs = String[]
    mh = create_time_series_for_model!(
        log, breadcrumbs, model_description,
        time_dimension, logging_policy,
    )
    return (log, mh)
end

function close_log(log::HDF5Log)
    if isvalid(log.group)
        close(log.group)
    end
    if !isnothing(log.fid) && isopen(log.fid)
        close(log.fid)
        log.fid = nothing
    end
    return nothing
end

function load_groups(group)

    # Files written before dimension groups were persisted have no group datasets. Missing
    # lets each caller retain its previous default behavior.
    if !haskey(group, "group_labels")
        return missing
    end
    if haskey(group, "groups_are_missing") && read(group["groups_are_missing"])
        return missing
    end

    group_labels = read(group["group_labels"])
    dimension_counts = read(group["group_dimension_counts"])
    dimension_labels = read(group["group_dimension_labels"])
    next_dimension = firstindex(dimension_labels)
    groups = Pair{String, Vector{String}}[]
    for (label, count) in zip(group_labels, dimension_counts)
        final_dimension = next_dimension + count - 1
        labels = String[dimension_labels[next_dimension:final_dimension]...]
        push!(groups, String(label) => labels)
        next_dimension = final_dimension + 1
    end
    return groups

end

function load_interpolator(group)

    # Older files did not record interpolation behavior. Missing asks the TimeSeries
    # constructor to continue deriving its default from the data type and discrete flag.
    if !haskey(group, "serialized_interpolator")
        return missing
    end

    bytes = Vector{UInt8}(read(group["serialized_interpolator"]))
    return deserialize_from_bytes(bytes)

end

function load_hdf5_timeseries(group)
    return TimeSeries(;
        title = read(group["title"]),
        time = load_hdf5_vector(group["time"]),
        data = load_hdf5_vector(group["data"]),
        time_dimension = Dimension(read(group["time_label"]), read(group["time_units"])),
        dimensions = [
            Dimension(label, units)
            for (label, units) in zip(read(group["labels"]), read(group["units"]))
        ],
        path = read(group["path"]),
        discrete = read(group["discrete"]),
        interpolator = load_interpolator(group),
        groups = load_groups(group),
    )
end

load_time_series_from_hdf5(fid, path) = load_hdf5_timeseries(fid[path])

function load_hdf5_constant(group)

    value_is_missing = haskey(group, "value_is_missing") &&
        read(group["value_is_missing"])
    value = if value_is_missing
        missing
    else
        constant_vector = load_hdf5_vector(group["value"])
        constant_vector[1]
    end

    is_description = haskey(group, "is_variable_description") &&
        read(group["is_variable_description"])
    if !is_description
        return value
    end

    value_type = deserialize_from_bytes(
        Vector{UInt8}(read(group["serialized_value_type"])),
    )
    dimensions = [
        Dimension(label, units)
        for (label, units) in zip(read(group["labels"]), read(group["units"]))
    ]
    return VariableDescription{value_type}(value;
        title = read(group["title"]),
        dimensions,
        groups = load_groups(group),
        interpolator = load_interpolator(group),
    )

end

function load_model_type(group, model_path)

    # Older files contain only the readable type string. The type is useful metadata, but
    # it is not required to inspect the recorded values, so unavailable definitions should
    # not make the rest of a log unusable.
    if !haskey(group, "serialized_type")
        return Missing
    end

    try
        bytes = Vector{UInt8}(read(group["serialized_type"]))
        type = deserialize_from_bytes(bytes)
        if !(type isa Type)
            error("The serialized model type produced a $(typeof(type)) value.")
        end
        return type
    catch err
        @warn "Could not restore the $model_path model type. Using Missing." exception = (
            err,
            catch_backtrace(),
        )
        return Missing
    end

end

function load_hdf5_model!(mhd, group, breadcrumbs)

    slug = isempty(breadcrumbs) ? "/" : join("/" * model for model in breadcrumbs)

    # HDF5 group iteration does not preserve the NamedTuple field order. New files record
    # it explicitly; the group keys retain the old behavior for files written previously.
    model_names = if haskey(group["names"], "models")
        read(group["names/models"])
    elseif haskey(group, "models")
        collect(keys(group["models"]))
    else
        String[]
    end

    # Construct the histories of the submodels.
    models = NamedTuple(
        Symbol(k) => load_hdf5_model!(mhd, group["models"][k], vcat(breadcrumbs, k))
        for k in model_names
    )

    # Figure out which times series is which kind of thing.
    constant_names = read(group["names/constants"])
    continuous_state_names = read(group["names/continuous_states"])
    discrete_state_names = read(group["names/discrete_states"])
    continuous_output_names = read(group["names/continuous_outputs"])
    discrete_output_names = read(group["names/discrete_outputs"])

    mh = ModelHistory(;
        type = load_model_type(group, slug),
        path = slug,
        constants = NamedTuple(
            Symbol(k) => load_hdf5_constant(group["constants"][k])
            for k in constant_names
        ),
        continuous_states = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k])
            for k in continuous_state_names
        ),
        discrete_states = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k])
            for k in discrete_state_names
        ),
        continuous_outputs = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k])
            for k in continuous_output_names
        ),
        discrete_outputs = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k])
            for k in discrete_output_names
        ),
        models,
    )

    # Save to the dictionary.
    mhd[slug] = mh

    # Return the tree.
    return mh

end

function load_hdf5_log(filename::AbstractString; path = "/log")

    fid = HDF5.h5open(filename)
    try

        # Older standalone logs stored the root model directly at the file root, so this is
        # a simple backwards-compatibility check to load those.
        if path == "/log" && !haskey(fid, path) && haskey(fid, "names")
            path = "/"
        end

        log, mh = load_hdf5_log(fid, path)

        # If there is no log, there's no point in keeping the HDF5 file open, so we'll close
        # it.
        own_log_file(log, fid)

        return (log, mh)

    catch

        close(fid)
        rethrow()

    end

end

function load_hdf5_log(group::HDF5.Group)

    # If the log was a NullLog to begin with or it wasn't saved, we can just return a
    # NullLog.
    if haskey(group, "is_null") && read(group["is_null"])
        return (SystemsOfSystems.Logs.NullLog(), nothing)
    end

    # Recursively load the ModelHistories, storing entires in the model history dictionary
    # by slug.
    mhd = OrderedDict{String, ModelHistory}()
    mh = load_hdf5_model!(mhd, group, String[])

    # Now an HDF5Log can store those.
    log = HDF5Log(nothing, HDF5.open_group(group, "."), mhd)

    # The user is required to close the log, but this is a helpful follow-up we can provide.
    finalizer(close_log, log)

    return (log, mh)

end

# TODO: A comment about why this function is necessary would be helpful. Couldn't the user
# simply call load_hdf5_log(parent[path]) instead of load_hdf5_log(parent, path)? How does
# this interface help anyone? Does closing the group when there's an error actually matter?
function load_hdf5_log(parent::Union{HDF5.File, HDF5.Group}, path::AbstractString)
    group = parent[path]
    try
        return load_hdf5_log(group)
    finally
        close(group)
    end
end

function save_time_series_to_hdf5(fid, path, ts::TimeSeries; kwargs...)

    # Set up the group and add the metadata.
    group = HDF5.create_group(fid, path)
    record_time_series_metadata(group, ts)

    # For the time and data, we'll use the copy_to_hdf5_vector to be totally consistent with
    # how these are created by HDF5Log.
    copy_to_hdf5_vector(group, "time", ts.time; kwargs...)

    # If we're logging an array and its dimensions are always exactly the same, we can
    # provide those to copy_to_hdf5_vector, which can store this much more efficiently.
    dims = nothing
    if eltype(ts.data) <: Array && !isempty(ts.data)
        first_dims = size(first(ts.data))
        if all(size(el) == first_dims for el in collect(ts.data))
            dims = first_dims
        end
    end
    copy_to_hdf5_vector(group, "data", ts.data; dims, kwargs...)

    return nothing

end

function save_mh_to_hdf5(group, mh; kwargs...)

    # Note: Every path is relative to this model's group, whether the log is standalone or
    # nested inside a whole-history file.

    group["type"] = string(mh.type)
    group["serialized_type"] = serialize_to_bytes(mh.type)

    # Constants may contain unsupported values. Keep the names of successful entries so
    # the loader can reconstruct the same ordering while omitting failed saves.
    constants_group = HDF5.create_group(group, "constants")
    names_group = HDF5.create_group(group, "names")
    saved_constants = String[]
    breadcrumbs = split(mh.path, '/'; keepempty = false)
    for (name, constant) in pairs(mh.constants)
        constant_group = HDF5.create_group(constants_group, string(name))
        try
            record_constant(constant_group, constant, breadcrumbs, string(name))
            push!(saved_constants, string(name))
        catch err
            trace = catch_backtrace()
            path = rstrip(mh.path, '/') * "/$name"
            message = "Failed to record the $path constant of type $(typeof(constant)) in " *
                "the HDF5 output file. Skipping."
            @warn message exception = (err, trace)
            HDF5.delete_object(constant_group)
        end
    end
    names_group["constants"] = saved_constants

    # States and outputs must all be saved; a failure here is an error.
    for f in (:continuous_states, :discrete_states, :continuous_outputs, :discrete_outputs)
        names_group[string(f)] = String[
            string(vn) for vn in fieldnames(typeof(getproperty(mh, f)))
        ]
        for (name, ts) in pairs(getproperty(mh, f))
            save_time_series_to_hdf5(group, "timeseries/$name", ts; kwargs...)
        end
    end

    # Preserve submodel order separately because HDF5 group iteration does not.
    names_group["models"] = String[string(name) for name in keys(mh.models)]
    for (name, smh) in pairs(mh.models)
        child_group = HDF5.create_group(group, "models/$name")
        save_mh_to_hdf5(child_group, smh; kwargs...)
    end

    return nothing

end

function save_log_to_hdf5(
    filename::AbstractString,
    log::AbstractLog;
    path = "/log",
    kwargs...,
)
    fid = history_file(log, filename)
    if isnothing(fid)
        HDF5.h5open(filename, "w") do output
            save_log_to_hdf5(output, path, log; kwargs...)
        end
    else
        save_log_to_hdf5(fid, path, log; kwargs...)
    end
    return nothing
end

function save_log_to_hdf5(
    parent::Union{HDF5.File, HDF5.Group},
    path::AbstractString,
    log::AbstractLog;
    kwargs...,
)
    group = storage_group(parent, path)
    try
        save_log_to_hdf5(group, log; kwargs...)
    finally
        close(group)
    end
    return nothing
end

include("sim_history.jl")

end
