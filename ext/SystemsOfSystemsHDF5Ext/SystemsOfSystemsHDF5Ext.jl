module SystemsOfSystemsHDF5Ext

using OrderedCollections: OrderedDict
import HDF5
using HDF5Vectors: create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector

using SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription
using SystemsOfSystems.Logs: ModelHistory, AbstractLogOptions, AbstractLog, HDF5LogOptions, create_time_series_for_model!

import SystemsOfSystems.Logs: create_log, create_time_series_for_var, record_model_description, close_log, load_hdf5_log, save_log_to_hdf5

"""
    HDF5Log(; fid, model_history_dict)

This acts like a BasicLog (stores all the same continuous and discrete states and outputs,
as well as constants and metadata), but the underlying storage is an HDF5 file. This
prevents the need for logs to be stored on disk -- critical for very long simulations. Note,
however, that this is much slower than BasicLog.

If you're just looking to have an HDF5 file artifact, it's faster to use a BasicLog and then
use `save_to_hdf5_log` when the simulation is over.

See [`HDF5LogOptions`](@ref) for more.
"""
mutable struct HDF5Log <: AbstractLog
    fid::Union{HDF5.File, Nothing}
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

function create_time_series_for_var(log::HDF5Log, breadcrumbs, var_name, var::VariableDescription{T}, time_dimension; discrete) where {T}
    el_type = figure_out_el_type(T)
    group_path = join("/models/" * el for el in breadcrumbs) * "/timeseries/" * var_name
    slug = join("/" * model for model in breadcrumbs) * "/" * var_name
    # println("Creating HDF5Vectors for $(var.title) at $group_path with type $el_type.")
    group = HDF5.create_group(log.fid, group_path)
    group["title"] = var.title
    group["time_label"] = time_dimension.label
    group["time_units"] = time_dimension.units
    group["labels"] = [dim.label for dim in var.dimensions]
    group["units"] = [dim.units for dim in var.dimensions]
    return TimeSeries(
        var.title,
        create_hdf5_vector(group, "time", Float64),
        create_hdf5_vector(group, "data", el_type),
        time_dimension,
        var.dimensions,
        slug,
        discrete,
    )
end
function create_time_series_for_var(log::HDF5Log, breadcrumbs, var_name, var::T, time_dimension; discrete) where {T}
    el_type = figure_out_el_type(T)
    group_path = join("/models/" * el for el in breadcrumbs) * "/timeseries/" * var_name
    slug = join("/" * model for model in breadcrumbs) * "/" * var_name
    # println("Creating HDF5Vectors at $group_path with type $el_type.")
    group = HDF5.create_group(log.fid, group_path)
    group["title"] = slug
    group["time_label"] = time_dimension.label
    group["time_units"] = time_dimension.units
    group["labels"] = String[]
    group["units"] = String[]
    return TimeSeries(
        slug,
        create_hdf5_vector(group, "time", Float64),
        create_hdf5_vector(group, "data", el_type),
        time_dimension,
        Dimension[], # TODO: Attempt to automatically list dimensions so they're not empty?
        slug,
        discrete,
    )
end

function record_constant(constant_group, v::VariableDescription{T}, breadcrumbs, name) where {T}
    constant_group["title"] = v.title
    type = figure_out_el_type(T)
    vec = create_hdf5_vector(constant_group, "value", type; chunk_length = 1)
    push!(vec, v.value)
    constant_group["labels"] = String[d.label for d in v.dimensions]
    constant_group["units"] = String[d.label for d in v.dimensions]
end
function record_constant(constant_group, v, breadcrumbs, name)
    constant_group["title"] = join("/" * el for el in breadcrumbs) * "/$name"
    vec = create_hdf5_vector(constant_group, "value", typeof(v); chunk_length = 1)
    push!(vec, v)
    constant_group["labels"] = String[]
    constant_group["units"] = String[]
end

function record_model_description(log::HDF5Log, breadcrumbs, md::ModelDescription)

    # Get the ground started.
    group_path = join("/models/" * el for el in breadcrumbs)
    if !isempty(group_path)
        group = HDF5.create_group(log.fid, group_path)
    else
        group = log.fid["/"] # This exists at creation.
    end

    # It's helpful to record what type generated this stuff. We can't actually load this as
    # a type, and it's not always available, but when it is, it seems like a helpful thing
    # to log.
    group["type"] = string(md.type)

    # We save the constants and use a try-catch, because we don't know how to save
    # everything a user might have as a constant to HDF5, and we'd rather just omit
    # constants than throw an error. We'll give a warning though.
    constants_group = HDF5.create_group(group, "constants")
    saved_constants = String[]
    for (k, v) in pairs(md.constants)
        constant_group = HDF5.create_group(constants_group, string(k))
        try
            record_constant(constant_group, v, breadcrumbs, k)
            push!(saved_constants, string(k))
        catch err
            p = join("/" * el for el in breadcrumbs) * "/$k"
            @warn "Failed to record the $p constant in the HDF5 output file. Skipping."
            HDF5.delete_group(constant_group)
        end
    end

    # We record the names of each type of thing. This helps us know if, say, "position" is a
    # state or output or constant.
    names_group_path = group_path * "/names"
    names_group = HDF5.create_group(log.fid, names_group_path)
    names_group["constants"] = saved_constants
    names_group["continuous_states"] = String[string(k) for k in keys(md.continuous_states)]
    names_group["discrete_states"] = String[string(k) for k in keys(md.discrete_states)]
    names_group["continuous_outputs"] = String[string(k) for k in keys(md.continuous_outputs)]
    names_group["discrete_outputs"] = String[string(k) for k in keys(md.discrete_outputs)]

    return nothing

end

function create_log(options::HDF5LogOptions, model_description, time_dimension)
    mkpath(dirname(options.filename))
    fid = HDF5.h5open(options.filename, "w")
    mhd = OrderedDict{String, ModelHistory}()
    log = HDF5Log(fid, mhd)
    finalizer(close_log, log) # Close the file when this goes out of scope.
    breadcrumbs = String[]
    mh = create_time_series_for_model!(log, breadcrumbs, model_description, time_dimension)
    return (log, mh)
end

function close_log(log::HDF5Log)
    if !isnothing(log.fid) && isopen(log.fid)
        close(log.fid)
        log.fid = nothing # TODO: What's the point of this?
    end
end

function load_hdf5_timeseries(group, breadcrumbs, var_name; discrete)
    slug = join("/" * model for model in breadcrumbs) * "/" * var_name
    ts = TimeSeries(
        read(group["title"]),
        load_hdf5_vector(group["time"]),
        load_hdf5_vector(group["data"]),
        Dimension(read(group["time_label"]), read(group["time_units"])),
        [Dimension(l, u) for (l, u) in zip(read(group["labels"]), read(group["units"]))],
        slug,
        discrete,
    )
    return ts
end

function load_hdf5_constant(group)
    constant_vector = load_hdf5_vector(group["value"]) # We store constants as 1-element vectors.
    return constant_vector[1]
end

function load_hdf5_model!(mhd, group, breadcrumbs)

    slug = isempty(breadcrumbs) ? "/" : join("/" * model for model in breadcrumbs)

    # Construct the histories of the submodels.
    models = if haskey(group, "models")
        NamedTuple(
            Symbol(k) => load_hdf5_model!(mhd, group["models"][k], vcat(breadcrumbs, k))
            for k in keys(group["models"])
        )
    else
        (;)
    end

    # Figure out which times series is which kind of thing.
    constant_names = read(group["names/constants"])
    continuous_state_names = read(group["names/continuous_states"])
    discrete_state_names = read(group["names/discrete_states"])
    continuous_output_names = read(group["names/continuous_outputs"])
    discrete_output_names = read(group["names/discrete_outputs"])

    mh = ModelHistory(;
        type = Missing, # The type is missing because we can't load it from an HDF5 file.
        path = slug,
        constants = NamedTuple(
            Symbol(k) => load_hdf5_constant(group["constants"][k])
            for k in constant_names
        ),
        continuous_states = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = false)
            for k in continuous_state_names
        ),
        discrete_states = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = true)
            for k in discrete_state_names
        ),
        continuous_outputs = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = false)
            for k in continuous_output_names
        ),
        discrete_outputs = NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = true)
            for k in discrete_output_names
        ),
        models,
    )

    # Save to the dictionary.
    mhd[slug] = mh

    # Return the tree.
    return mh

end

"""
    load_hdf5_log(filename::AbstractString)

Loads an HDF5Log from the given HDF5 file. Returns (log, model_history).
"""
function load_hdf5_log(filename::AbstractString)
    fid = HDF5.h5open(filename)
    mhd = OrderedDict{String, ModelHistory}()
    breadcrumbs = String[]
    mh = load_hdf5_model!(mhd, fid["/"], breadcrumbs)
    log = HDF5Log(fid, mhd)
    return (log, mh)
end

function save_ts_to_hdf5(fid, breadcrumbs, var_name, ts::TimeSeries; kwargs...)

    # Set up the group and add the metadata.
    group_path = join("/models/" * el for el in breadcrumbs) * "/timeseries/" * var_name
    group = HDF5.create_group(fid, group_path)
    group["title"] = ts.title
    group["time_label"] = ts.time_dimension.label
    group["time_units"] = ts.time_dimension.units
    group["labels"] = [dim.label for dim in ts.dimensions]
    group["units"] = [dim.units for dim in ts.dimensions]

    # For the time and data, we'll use the copy_to_hdf5_vector to be totally consistent with
    # how these are created by HDF5Log.
    copy_to_hdf5_vector(group, "time", ts.time; kwargs...)

    # If we're logging an array and it's dimensions are always exactly the same, we can
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

function save_mh_to_hdf5(fid, mh, breadcrumbs; kwargs...)

    # Set up the path to here, like /models/subsystem1/models/subsubsystem2.
    this_path = join("/models/" * el for el in breadcrumbs)

    # Record the type just for users to look at. We don't load this string as if it were a
    # type or use it in any way.
    fid["$this_path/type"] = string(mh.type)

    # Save the constants. This may fail since the user may have all kinds of constants that
    # we don't know how to log, so make sure we record only the successful ones.
    constants_path = this_path * "/constants/"
    constants_group = HDF5.create_group(fid, constants_path)
    saved_constants = String[]
    for (name, constant) in pairs(mh.constants)
        constant_group = HDF5.create_group(constants_group, string(name))
        try
            record_constant(constant_group, constant, breadcrumbs, string(name))
            push!(saved_constants, string(name))
        catch
            p = join("/" * el for el in breadcrumbs) * "/$name"
            @warn "Failed to record the $p constant in the HDF5 output file. Skipping."
            HDF5.delete_group(constant_group)
        end
    end
    fid["$this_path/names/constants"] = saved_constants

    # Now save all states and outputs. We don't use a try-catch here. We need to be able to
    # save everything, or it's a good idea to throw an error.
    for f in (:continuous_states, :discrete_states, :continuous_outputs, :discrete_outputs)
        fid["$this_path/names/$f"] = String[
            string(vn) for vn in fieldnames(typeof(getproperty(mh, f)))
        ]
        for (name, ts) in pairs(getproperty(mh, f))
            save_ts_to_hdf5(fid, breadcrumbs, string(name), ts; kwargs...)
        end
    end

    # Do the submodels.
    for (name, smh) in pairs(mh.models)
        save_mh_to_hdf5(fid, smh, vcat(breadcrumbs, string(name)); kwargs...)
    end

    return nothing

end

"""
    save_log_to_hdf5(filename::AbstractString, log::AbstractLog; kwargs...)

Saves any other type of log in the same format used by HDF5Log so that it can be loaded as
an HDF5Log or loaded outside of Julia. Returns nothing.

Any additional keyword arguments are passed through to `HDF5Vectors.copy_to_hdf5_vector`.
These can allow better control over how the data is stored in the HDF5 file. See that
package for details.
"""
function save_log_to_hdf5(filename::AbstractString, log::AbstractLog; kwargs...)
    HDF5.h5open(filename, "w") do fid
        breadcrumbs = String[]
        save_mh_to_hdf5(fid, log["/"], breadcrumbs; kwargs...)
    end
    return nothing
end

end
