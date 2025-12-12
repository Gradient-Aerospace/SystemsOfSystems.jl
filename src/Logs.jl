"""
TODO
"""
module Logs

using ..SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription
using OrderedCollections: OrderedDict

################
# ModelHistory #
################

"""
TODO
"""
@kwdef struct ModelHistory
    path::String
    constants::NamedTuple
    continuous_states::NamedTuple # where all the elements are TimeSeries
    discrete_states::NamedTuple
    continuous_outputs::NamedTuple
    discrete_outputs::NamedTuple
    models::NamedTuple
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

function Base.getindex(mh::ModelHistory, key::String)
    return getindex(mh, Symbol(key))
end
function Base.getindex(mh::ModelHistory, key::Symbol)
    if haskey(mh.constants, key)
        return mh.constants[key] # Could be any time.
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

# function show_all(io::IO, mh::ModelHistory, slug = "/"; indent = 2)
#     println("  "^indent * slug)
# end

###############
# AbstractLog #
###############

export AbstractLogOptions, AbstractLog, create_log, close_log, gather_all_time_series

"""
TODO
"""
abstract type AbstractLogOptions end

"""
TODO

Functions:
* create_log
* close_log
* getindex, setindex!, keys, pairs
"""
abstract type AbstractLog end

# This isn't used by the BasicLog, but it lets the HDF5Log record extra details.
function record_model_description(log::AbstractLog, breadcrumbs, md)
    nothing
end

# "Sets" include continuous states, discrete outputs, etc.
function create_time_series_for_set(log::AbstractLog, breadcrumbs, set, time_dimension; discrete = true)
    return NamedTuple(
        f => create_time_series_for_var(log, breadcrumbs, string(f), v, time_dimension; discrete)
        for (f, v) in pairs(set)
    )
end

function create_time_series_for_model!(log::AbstractLog, breadcrumbs, md::ModelDescription, time_dimension)

    slug = isempty(breadcrumbs) ? "/" : join("/" * el for el in breadcrumbs)

    # Record any extra stuff.
    record_model_description(log, breadcrumbs, md)

    # Create the time histories.
    mh = ModelHistory(;
        path = slug,
        constants = md.constants,
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
    log[slug] = mh

    return mh

end

"""
TODO
"""
function create_log(options::AbstractLogOptions, model_description, time_dimension)
    error("No `create_log` implementation exists for $(typeof(options)).")
end

"""
TODO
"""
function close_log(::AbstractLog)
    return nothing
end

gather_all_time_series(log::AbstractLog) = gather_all_time_series(log["/"])

# TODO: Consider making a getindex that breaks apart a single string into model and var.
# This implies that subtypes of AbstractLog would implement get_model_history instead of
# get_index. An alternative is to implement `get_dict` and let AbstractLog take care of all
# of the dict-like interface.
# function Base.getindex(log::AbstractLog, k)
#     if contains(k, ':')
#         parts = split(k, ':')
#         @assert length(parts) == 2 "Expected key like /model/path:var_name but got \"$k\"."
#         slug = parts[1]
#         var_name = parts[2]
#         return get_model_history(log, slug)[var_name]
#     else
#         return get_model_history(log, k)
#     end
# end

############
# BasicLog #
############

export BasicLogOptions

"""
TODO
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

function create_time_series_for_var(::BasicLog, breadcrumbs, var_name, var::VariableDescription{T}, time_dimension; discrete = true) where {T}
    return TimeSeries(
        var.title,
        Float64[],
        T[],
        time_dimension,
        var.dimensions,
        join("/" * el for el in breadcrumbs),
        discrete,
    )
end
function create_time_series_for_var(::BasicLog, breadcrumbs, var_name, var::T, time_dimension; discrete = true) where {T}
    return TimeSeries(
        join("/" * el for el in breadcrumbs), # Let the slug be the title.
        Float64[],
        T[],
        time_dimension,
        Dimension[], # TODO: Attempt to automatically list dimensions?
        join("/" * el for el in breadcrumbs),
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
TODO
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

export HDF5LogOptions, load_hdf5_log

"""
TODO
"""
@kwdef struct HDF5LogOptions <: AbstractLogOptions
    filename::String
end

"""
TODO
"""
function load_hdf5_log(filename)
    error("Please import the HDF5 package to use HDF5 log functionality like `load_hdf5_log`.")
end

end
