"""
This module contains the different log types: `BasicLog` (the default), `NullLog` (doesn't
log), and `HDF5Log` (import HDF5 for this one to work).
"""
module Logs

using ..SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription
using OrderedCollections: OrderedDict

################
# ModelHistory #
################

export ModelHistory

"""
This stores the time history of a single model, including its discrete and continuous states
and outputs, as well as constants, the "path" to this model, and the model histories for its
sub-models.
"""
@kwdef mutable struct ModelHistory{
    CT  <: NamedTuple,
    XCT <: NamedTuple,
    YCT <: NamedTuple,
    XDT <: NamedTuple,
    YDT <: NamedTuple,
    MT  <: NamedTuple,
}
    type::Type
    path::String
    constants::CT # all elements are raw values
    continuous_states::XCT # all elements are TimeSeries
    discrete_states::XDT
    continuous_outputs::YCT
    discrete_outputs::YDT
    models::MT
end

function Base.keys(mh::ModelHistory)
    return vcat(
        collect(keys(mh.constants)),
        collect(keys(mh.continuous_states)),
        collect(keys(mh.discrete_states)),
        collect(keys(mh.continuous_outputs)),
        collect(keys(mh.discrete_outputs)),
        collect(keys(mh.models)),
    )
end

function Base.values(mh::ModelHistory)
    return vcat(
        collect(mh.constants),
        collect(mh.continuous_states),
        collect(mh.discrete_states),
        collect(mh.continuous_outputs),
        collect(mh.discrete_outputs),
        collect(mh.models),
    )
end

function Base.pairs(mh::ModelHistory)
    return zip(keys(mh), values(mh))
end

function Base.getindex(mh::ModelHistory, key::AbstractString)
    return getindex(mh, Symbol(key))
end
function Base.getindex(mh::ModelHistory, key::Symbol)
    if haskey(mh.constants, key)
        return mh.constants[key] # Could be any type.
    elseif haskey(mh.continuous_states, key)
        return mh.continuous_states[key] # TimeSeries
    elseif haskey(mh.discrete_states, key)
        return mh.discrete_states[key] # TimeSeries
    elseif haskey(mh.continuous_outputs, key)
        return mh.continuous_outputs[key] # TimeSeries
    elseif haskey(mh.discrete_outputs, key)
        return mh.discrete_outputs[key] # TimeSeries
    elseif haskey(mh.models, key)
        return mh.models[key] # ModelHistory
    end
    error("The ModelHistory has no $key key. Available keys: $(keys(mh)).")
end

function show_container_keys(io, name, container)
    if isempty(container)
        # println(io, "  $name: (none)")
    else
        println(io, "  $name:")
        for (k, v) in pairs(container)
            println(io, "    $k => $(typeof(v))")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", mh::ModelHistory)
    println(io, "ModelHistory for $(mh.path) with the following contents:")
    println(io, "  type: ", mh.type)
    show_container_keys(io, "constants", mh.constants)
    show_container_keys(io, "continuous_states", mh.continuous_states)
    show_container_keys(io, "discrete_states", mh.discrete_states)
    show_container_keys(io, "continuous_outputs", mh.continuous_outputs)
    show_container_keys(io, "discrete_outputs", mh.discrete_outputs)
    show_container_keys(io, "models", mh.models)
end

function gather_all_time_series!(tss, mh::ModelHistory, slug)
    for (k, ts) in pairs(mh.continuous_states)
        tss["$slug:$k"] = ts
    end
    for (k, ts) in pairs(mh.discrete_states)
        tss["$slug:$k"] = ts
    end
    for (k, ts) in pairs(mh.continuous_outputs)
        tss["$slug:$k"] = ts
    end
    for (k, ts) in pairs(mh.discrete_outputs)
        tss["$slug:$k"] = ts
    end
    for (k, m) in pairs(mh.models)
        gather_all_time_series!(tss, m, slug * "/$k")
    end
end

function gather_all_time_series(mh::ModelHistory)
    tss = OrderedDict{String, TimeSeries}()
    gather_all_time_series!(tss, mh, "")
    return tss
end

###############
# AbstractLog #
###############

export AbstractLogOptions, AbstractLog, create_log, close_log, gather_all_time_series

"""
A set of options for setting up the log of the appropriate type.
"""
abstract type AbstractLogOptions end

"""
All AbstractLog types are expected to obey this interface to work with simulations in
SystemsOfSystems.

Functions:
* `create_log`
* `close_log`
* `getindex`, `setindex!`, `keys`, `values`, `pairs`
"""
abstract type AbstractLog end

# This isn't used by the BasicLog, but it lets the HDF5Log record extra details.
function record_model_description(log::AbstractLog, breadcrumbs, md)
    nothing
end

# "Sets" include continuous states, discrete outputs, etc.
function create_time_series_for_set(
    log::AbstractLog, breadcrumbs, set, time_dimension;
    discrete = true,
)

    # Make a named tuple containing the TimeSeries for all logged signals of this set.
    return NamedTuple(
        f => create_time_series_for_var(
            log, breadcrumbs, string(f), v, time_dimension; discrete,
        )
        for (f, v) in pairs(set)
    )

end

function create_time_series_for_model!(
    log::AbstractLog,
    breadcrumbs,
    md::ModelDescription,
    time_dimension,
)

    # Form this model's path.
    path = isempty(breadcrumbs) ? "/" : join("/" * el for el in breadcrumbs)

    # Record any extra stuff.
    record_model_description(log, breadcrumbs, md)

    # Create the time histories.
    mh = ModelHistory(;
        type = md.type,
        path = path,
        constants = md.constants, # TODO: Should this "decorate" the constants as VariableDescriptions, like we add decorators for the TimeSeries, below?
        continuous_states = create_time_series_for_set(log, breadcrumbs, md.continuous_states, time_dimension; discrete = false),
        # TODO: Record derivatives too.
        discrete_states = create_time_series_for_set(log, breadcrumbs, md.discrete_states, time_dimension; discrete = true),
        continuous_outputs = create_time_series_for_set(log, breadcrumbs, md.continuous_outputs, time_dimension; discrete = false),
        discrete_outputs = create_time_series_for_set(log, breadcrumbs, md.discrete_outputs, time_dimension; discrete = true),
        models = NamedTuple(
            f => create_time_series_for_model!(log, vcat(breadcrumbs, string(f)), m, time_dimension)
            for (f, m) in pairs(md.models)
        )
    )

    # Put it in the dictionary of time histories.
    log[path] = mh

    return mh

end

"""
    create_log(options::AbstractLogOptions, model_description, time_dimension)

Creates a log type and model history for the given `model_description` and `time_dimension`.
Returns the log and model history as a tuple.
"""
function create_log(options::AbstractLogOptions, model_description, time_dimension)
    error("No `create_log` implementation exists for $(typeof(options)).")
end

"""
    close_log(::AbstractLog)

If there are any resources open for the given log, this closes them (which may make some
logs non-operational).
"""
function close_log(::AbstractLog)
    return nothing
end

gather_all_time_series(log::AbstractLog) = gather_all_time_series(log["/"])

function Base.show(io::IO, mime::MIME"text/plain", log::AbstractLog)
    println(io, "Model Histories:")
    slugs = sort(collect(keys(log)))
    for slug in slugs
        println(io, "  " * slug)
    end
end

############
# BasicLog #
############

export BasicLogOptions

"""
There are no options for a `BasicLog`, so this is an empty structure.
"""
struct BasicLogOptions <: AbstractLogOptions end

"""
    BasicLog

This logs all sim results in arrays. It's the simplest and fastest log, but for sims with
too much output to fit in RAM, a disk-based log (like HDF5Log) is a better choice.
"""
struct BasicLog <: AbstractLog
    model_history_dict::OrderedDict{String, ModelHistory}
end

Base.setindex!(log::BasicLog, mh, slug) = (log.model_history_dict[slug] = mh)
Base.getindex(log::BasicLog, k) = log.model_history_dict[k]
Base.keys(log::BasicLog) = keys(log.model_history_dict)
Base.values(log::BasicLog) = values(log.model_history_dict)
Base.pairs(log::BasicLog) = pairs(log.model_history_dict)

function create_time_series_for_var(
    ::BasicLog,
    breadcrumbs,
    var_name,
    var::VariableDescription{T},
    time_dimension;
    discrete = true
) where {T}

    model_path = join("/" * el for el in breadcrumbs)
    signal_path = model_path * "/" * var_name

    return TimeSeries(;
        var.title,
        time = Float64[],
        data = T[],
        time_dimension,
        var.dimensions,
        path = signal_path,
        discrete,
        var.interpolator,
        var.groups,
    )

end

function create_time_series_for_var(::BasicLog, breadcrumbs, var_name, var::T, time_dimension; discrete = true) where {T}
    return TimeSeries(;
        title = join("/" * el for el in breadcrumbs), # Let the slug be the title.
        time = Float64[],
        data = T[],
        time_dimension,
        path = join("/" * el for el in breadcrumbs),
        discrete,
    )
end

function create_log(::BasicLogOptions, model_description, time_dimension)
    log = BasicLog(OrderedDict{String, ModelHistory}())
    breadcrumbs = String[]
    mh = create_time_series_for_model!(log, breadcrumbs, model_description, time_dimension)
    return (log, mh)
end

###########
# NullLog #
###########

export NullLogOptions

"""
There are no options for a `NullLog`, so this is an empty structure.
"""
struct NullLogOptions <: AbstractLogOptions end

"""
    NullLog

This doesn't log anything. It's how you turn logging off.
"""
struct NullLog <: AbstractLog end

Base.setindex!(log::NullLog, mh, slug) = error("A NullLog holds no data.")
Base.getindex(log::NullLog, k) = error("A NullLog holds no data.")
Base.keys(log::NullLog) = ()
Base.values(log::NullLog) = ()
Base.pairs(log::NullLog) = Vector{Pair}[]

function create_log(::NullLogOptions, model_description, time_dimension)
    return (NullLog(), nothing)
end
function create_log(::Nothing, model_description, time_dimension)
    return (NullLog(), nothing)
end

###########
# HDF5Log #
###########

export HDF5LogOptions, load_hdf5_log, save_log_to_hdf5, save_time_series_to_hdf5

"""
    HDF5LogOptions(; filename)

The HDF5Log acts like a BasicLog (stores all the same continuous and discrete states and
outputs, as well as constants and metadata), but the underlying storage is an HDF5 file.
This prevents the need for logs to be stored on disk -- critical for very long simulations.
Note, however, that this is much slower than BasicLog.

This structure contains the options for the HDF5Log, consisting only of a filename.

If you're just looking to have an HDF5 file artifact, it's faster to use a BasicLog and then
use `save_log_to_hdf5` when the simulation is over.
"""
@kwdef struct HDF5LogOptions <: AbstractLogOptions
    filename::String
end

"""
    load_hdf5_log(filename)

Loads a log from an HDF5 file.
"""
function load_hdf5_log(filename)
    error("Please import the HDF5Vectors package to use HDF5 log functionality like `load_hdf5_log`.")
end

"""
    save_log_to_hdf5(filename, log)

Saves a log to an HDF5 file in the same format used by the HDF5Log.
"""
function save_log_to_hdf5(filename, log)
    error("Please import the HDF5Vectors package to use HDF5 log functionality like `save_log_to_hdf5`.")
end

function save_time_series_to_hdf5(args...; kwargs...)
    error("Please import the HDF5Vectors package to use HDF5 log functionality like `save_time_series_to_hdf5`.")
end

end
