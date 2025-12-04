module SystemsOfSystemsHDF5Ext

# println("Loading SystemsOfSystemsHDF5Ext")

using OrderedCollections: OrderedDict
import HDF5
using HDF5Vectors: create_hdf5_vector, load_hdf5_vector

using SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription
using SystemsOfSystems.Logs: ModelHistory, AbstractLogOptions, AbstractLog, HDF5LogOptions, create_time_series_for_model!

import SystemsOfSystems.Logs: create_log, create_time_series_for_var, record_model_description, close_log, load_hdf5_log

"""
TODO
"""
mutable struct HDF5Log <: AbstractLog
    fid::Union{HDF5.File, Nothing}
    model_history_dict::OrderedDict{String, ModelHistory}
end

# TODO: Should we check that the file is open and give a friendly error if not?
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

function record_model_description(log::HDF5Log, breadcrumbs, md::ModelDescription)
    group_path = join("/models/" * el for el in breadcrumbs)
    if !isempty(group_path)
        group = HDF5.create_group(log.fid, group_path)
    else
        group = log.fid["/"] # This exists at creation.
    end
    constants_group = HDF5.create_group(group, "constants")
    for (k, v) in pairs(md.constants)
        constant_group = HDF5.create_group(constants_group, string(k))
        if v isa VariableDescription
            constant_group["title"] = v.title
            vec = create_hdf5_vector(constant_group, "value", typeof(v.value); chunk_length = 1)
            push!(vec, v.value)
            constant_group["labels"] = String[d.label for d in v.dimensions]
            constant_group["units"] = String[d.label for d in v.dimensions]
        else
            constant_group["title"] = join(["/" * el for el in breadcrumbs], string(k))
            vec = create_hdf5_vector(constant_group, "value", typeof(v.value); chunk_length = 1)
            push!(vec, v)
            constant_group["labels"] = String[]
            constant_group["units"] = String[]
        end
    end
    group["type"] = string(md.type)
    names_group_path = group_path * "/names"
    names_group = HDF5.create_group(log.fid, names_group_path)
    names_group["constants"] = String[string(k) for k in keys(md.constants)]
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
    close(log.fid)
    log.fid = nothing # TODO: What's the point of this?
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

    mh = ModelHistory(
        slug,
        NamedTuple(
            Symbol(k) => load_hdf5_constant(group["constants"][k])
            for k in constant_names
        ),
        NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = false)
            for k in continuous_state_names
        ),
        NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = true)
            for k in discrete_state_names
        ),
        NamedTuple(
            Symbol(k) => load_hdf5_timeseries(group["timeseries"][k], breadcrumbs, k; discrete = false)
            for k in continuous_output_names
        ),
        NamedTuple(
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
TODO
"""
function load_hdf5_log(filename::AbstractString)
    fid = HDF5.h5open(filename)
    mhd = OrderedDict{String, ModelHistory}()
    breadcrumbs = String[]
    mh = load_hdf5_model!(mhd, fid["/"], breadcrumbs)
    log = HDF5Log(fid, mhd)
    return (log, mh)
end

end
