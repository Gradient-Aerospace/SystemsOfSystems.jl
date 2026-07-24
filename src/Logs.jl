"""
This module contains the different log types: `BasicLog` (the default), `NullLog` (which
disables logging), and `HDF5Log` (available after importing HDF5Vectors).
"""
module Logs

using OrderedCollections: OrderedDict
using ..SystemsOfSystems: TimeSeries, Dimension, VariableDescription, ModelDescription,
    Samplers
using ..LoggingPolicies: AbstractLoggingPolicy, get_model_logging_policy,
    get_sampler, get_variable_set, is_variable_in_set,
    AllPassLoggingPolicy

###################
# Sampling Groups #
###################

# Models assigned the same sampler share one of these mutable decisions. The root logging
# entry point evaluates the sampler once at each opportunity, after which every matching
# model reads these plain booleans without repeating the trigger calculation.
mutable struct SamplingGroup{ST <: Samplers.AbstractSampler}
    sampler::ST
    log_states::Bool
    snapshot_states::Bool # Log every state, rather than only sparse state changes.
    log_outputs::Bool
end

SamplingGroup(sampler::Samplers.AbstractSampler) =
    SamplingGroup(sampler, false, false, false)

function update_sampling_group!(t, group::SamplingGroup)

    # Snapshotting refines state logging; it cannot independently enable state logging when
    # a custom directive has disabled it.
    directive = Samplers.get_sampling_directive(t, group.sampler)
    group.log_states = Samplers.should_log_states(directive)
    group.snapshot_states =
        group.log_states && Samplers.should_snapshot_states(directive)
    group.log_outputs = Samplers.should_log_outputs(directive)
    return nothing

end

function get_sampling_group!(sampling_groups, sampler)

    # Broad regex rules return the same sampler for many model paths. Immutable built-in
    # samplers with identical fields are also `===`, while mutable custom samplers share a
    # decision only when the user supplies the same object.
    for group in sampling_groups
        if group.sampler === sampler
            return group
        end
    end

    group = SamplingGroup(sampler)
    push!(sampling_groups, group)
    return group

end

function collect_sampling_groups_in_subtree(sampling_group, models)

    # Begin with this model's group, then append each distinct group used below it. Identity
    # is appropriate because get_sampling_group! has already canonicalized shared samplers.
    #
    # This is setup-time work, so use a growable array rather than repeatedly constructing
    # longer tuple types in the loop. The final splat recovers a concretely typed tuple for
    # allocation-free iteration in the simulation loop.
    sampling_groups = Any[sampling_group]
    for model in models
        for group in model.sampling_groups_in_subtree
            if !any(existing === group for existing in sampling_groups)
                push!(sampling_groups, group)
            end
        end
    end

    return tuple(sampling_groups...)

end

################
# ModelHistory #
################

"""
    ModelHistory

Store the history and logging configuration for one model.

The named-tuple fields contain the model's constants, state and output time series, and
recursive submodel histories. `path` identifies the model, using `/` for the root. `sampler`
decides which simulation-loop samples record this model.

A model logging policy may omit constants and time-series fields from these named tuples.

`ModelHistory` is mutable to give large, recursively parameterized histories reference
semantics. Its fields are established during log construction and are not normally
reassigned; samples are appended to the contained time series.

The internal `sampling_group` stores this model's current sampling decision. Models with
the same sampler share a group, so its trigger is evaluated once per logging opportunity.
`sampling_groups_in_subtree` lets the recursive logger reject inactive subtrees without
giving a parent sampler semantic control over its children.
"""
@kwdef mutable struct ModelHistory{ # This is mutable only to put it on the heap.
    CT  <: NamedTuple,
    XCT <: NamedTuple,
    XDT <: NamedTuple,
    YCT <: NamedTuple,
    YDT <: NamedTuple,
    MT  <: NamedTuple,
    ST   <: Samplers.AbstractSampler,
    SGT,
    SGST <: Tuple,
}
    type::Type
    path::String
    constants::CT # all elements are raw values
    continuous_states::XCT # all elements are TimeSeries
    discrete_states::XDT
    continuous_outputs::YCT
    discrete_outputs::YDT
    models::MT
    sampler::ST = Samplers.CompleteSampler()
    sampling_group::SGT = nothing # Shared current decision for this model.
    sampling_groups_in_subtree::SGST = () # Distinct decisions at or below this model.
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
function record_model_description(log::AbstractLog, breadcrumbs, md, variable_set)
    nothing
end

# TODO: Should this "decorate" the constants as VariableDescriptions, like we add decorators
# for the TimeSeries, below?
function store_constants(constants, variable_set)
    return NamedTuple(
        f => v
        for (f, v) in pairs(constants) if is_variable_in_set(f, variable_set)
    )
end

# "Sets" include continuous states, discrete outputs, etc.
function create_time_series_for_set(
    log::AbstractLog, breadcrumbs, set, variable_set, time_dimension;
    discrete = true,
)

    # Make a named tuple containing the TimeSeries for all logged signals of this set.
    return NamedTuple(
        f => create_time_series_for_var(
            log, breadcrumbs, string(f), v, time_dimension; discrete,
        )
        for (f, v) in pairs(set) if is_variable_in_set(f, variable_set)
    )

end

function create_time_series_for_model!(
    log::AbstractLog,
    breadcrumbs,
    md::ModelDescription,
    time_dimension,
    logging_policy::AbstractLoggingPolicy,
    sampling_groups = Any[],
)

    # Form this model's path.
    model_path = isempty(breadcrumbs) ? "/" : join("/" * el for el in breadcrumbs)

    # See what should be logged (variable set) and when it should be logged (sampler).
    model_logging_policy = get_model_logging_policy(logging_policy, model_path)
    variable_set = get_variable_set(model_logging_policy)
    sampler = get_sampler(model_logging_policy)
    sampling_group = get_sampling_group!(sampling_groups, sampler)

    # Record any extra stuff.
    record_model_description(log, breadcrumbs, md, variable_set)

    # Build the child histories first so this model can cache every distinct sampling group
    # used in its subtree.
    models = NamedTuple(
        f => create_time_series_for_model!(
            log, vcat(breadcrumbs, string(f)), m,
            time_dimension, logging_policy, sampling_groups,
        )
        for (f, m) in pairs(md.models)
    )
    sampling_groups_in_subtree =
        collect_sampling_groups_in_subtree(sampling_group, models)

    # Create the time histories.
    mh = ModelHistory(;
        type = md.type,
        path = model_path,
        constants = store_constants(md.constants, variable_set),
        continuous_states = create_time_series_for_set(
            log, breadcrumbs, md.continuous_states, variable_set, time_dimension;
            discrete = false,
        ),
        # TODO: Record derivatives too.
        discrete_states = create_time_series_for_set(
            log, breadcrumbs, md.discrete_states, variable_set, time_dimension;
            discrete = true,
        ),
        continuous_outputs = create_time_series_for_set(
            log, breadcrumbs, md.continuous_outputs, variable_set, time_dimension;
            discrete = false,
        ),
        discrete_outputs = create_time_series_for_set(
            log, breadcrumbs, md.discrete_outputs, variable_set, time_dimension;
            discrete = true,
        ),
        models,
        sampler = sampler,
        sampling_group,
        sampling_groups_in_subtree,
    )

    # Put it in the dictionary of time histories.
    log[model_path] = mh

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
    BasicLogOptions(; logging_policy = AllPassLoggingPolicy())

Configure an in-memory `BasicLog`.

`logging_policy` assigns a model logging policy to every model, by path. The default
`AllPassLoggingPolicy` logs all variables of all models on all samples.
"""
@kwdef struct BasicLogOptions <: AbstractLogOptions
    logging_policy::AbstractLoggingPolicy = AllPassLoggingPolicy()
end

"""
    BasicLog

Stores time histories for model variables in arrays. This is the simplest and fastest log,
but an HDF5 log is a better choice when the selected history is too large to fit in RAM.
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

    return TimeSeries(;
        var.title,
        time = Float64[],
        data = T[],
        time_dimension,
        var.dimensions,
        path = model_path,
        discrete,
        var.interpolator,
        var.groups,
    )

end

function create_time_series_for_var(
    ::BasicLog,
    breadcrumbs,
    var_name,
    var::T,
    time_dimension;
    discrete = true,
) where {T}

    model_path = join("/" * el for el in breadcrumbs)
    signal_path = model_path * "/" * var_name

    return TimeSeries(;
        title = signal_path,
        time = Float64[],
        data = T[],
        time_dimension,
        path = model_path,
        discrete,
    )

end

function create_log(options::BasicLogOptions, model_description, time_dimension)
    log = BasicLog(OrderedDict{String, ModelHistory}())
    logging_policy = options.logging_policy
    breadcrumbs = String[]
    mh = create_time_series_for_model!(
        log, breadcrumbs, model_description,
        time_dimension, logging_policy,
    )
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
    HDF5LogOptions(; filename, logging_policy)
    HDF5LogOptions(filename)

Configure an HDF5-backed log written to `filename`.

`logging_policy` assigns a model logging policy to every model, by path. The default
`AllPassLoggingPolicy` logs all variables of all models on all samples.

An HDF5 log records the same selected continuous and discrete states, outputs, constants,
and metadata as a `BasicLog`, but stores time-series data on disk. This supports histories
that would not fit in RAM, at the cost of slower logging.

If the selected history fits in memory and only the final artifact needs to be HDF5, it is
faster to use a `BasicLog` and call `save_log_to_hdf5` after simulation.
"""
@kwdef struct HDF5LogOptions <: AbstractLogOptions
    filename::String
    logging_policy::AbstractLoggingPolicy = AllPassLoggingPolicy()
end

# For backwards compatibility.
HDF5LogOptions(filename) = HDF5LogOptions(; filename)

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
