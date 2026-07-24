module SystemsOfSystems

# Running simulations
export initialize, simulate, SimOptions, Schedules, Solvers, Hooks, Logs, Resources

# Model descriptions
export ModelDescription, VariableDescription, RandomVariableDescription,
    RatesOutput, UpdatesOutput, on_triggering,
    OutputFile, Resource

# Utilities
export Dimension, TimeSeries,
    BranchingSeed, branch,
    AbstractTimeSeriesInterpolator, SampleAndHold, LinearInterpolation,
    ContinuousWhiteNoise, DiscreteWhiteNoise,
    AbstractSchedule, RegularSchedule, OffsetRegularSchedule,
    is_triggering, next_trigger_time,
    KEEP_T_NEXT, NO_T_NEXT,
    is_regular_step_triggering, next_regular_time,
    Samplers, LoggingPolicies

using Random: Xoshiro, randn

include("SimulationTimes.jl")
using .SimulationTimes

include("Schedules.jl")
using .Schedules

include("TimeSeries.jl")
using .TimeSeriesStuff

include("BranchingSeeds.jl")
using .BranchingSeeds

include("Resources.jl")
using .Resources

include("Hooks.jl")

# We could move this and the LoggingPolicies include inside the Logs module, but we leave
# them here so they are more easily accessible to users (Samplers.<whatever> instead of
# Logs.Samplers.<whatever>).
include("Samplers.jl")
using .Samplers

include("LoggingPolicies.jl")
using .LoggingPolicies

#########################
# User Function Outputs #
#########################

"""
This is the output expected from the `init_fcn` provided to `simulate`. It describes the
elements of the model, including:

* `type::Type`: The type that should be used when constructing the model (or Nothing to use
  a named tuple. The type should accept keyword arguments for the variables, below.
* `constants`: A named tuple of each constant the model should hold
* `continuous_states`: A named tuple of each of the continuous states in the model
* `discrete_states`: A named tuple of each of the discrete states in the model
* `continuous_outputs`: A named tuple of each of the continuous outputs in the model
* `discrete_outputs`: A named tuple of each of the continuous outputs in the model
* `continuous_random_variables`: A named tuple of each of the continuous random variables in
   the model. Each element can be a function mapping `(rng, t_last, t_next)` to a value, or
   a `RandomVariableDescription`.
* `discrete_random_variables`: A named tuple of each of the discrete random variables in the
  model. Each element can be a function mapping `(rng, t)` to a value, or a
  `RandomVariableDescription`.
* `schedules`: A named tuple of declarative `AbstractSchedule` values. Each named schedule
  is exposed on the constructed model like a constant, while also telling the simulation
  which exact times must become accepted samples.
* `models`: A named tuple containing the ModelDescription of each submodel.
* `resources`: A named tuple containing a `Resources.AbstractResource`, for opening files
  or creating connections that need to be closed when the simulation is over.
* `t_next`: The next sim time at which the model requests that the integrator stop. The
  integrator will step no later than this time, but may step earlier. It defaults to
  `NO_T_NEXT`, meaning the model has no finite scheduled event.

For the constants, states, and outputs, the value corresponding with each field can either
be a raw value (e.g., 6.) or a `VariableDescription`, such as:

```
VariableDescription(
    6;
    title = "Object Mass",
    dimensions = ["m" => "kg",],
)
```
"""
struct ModelDescription
    type
    constants
    continuous_states
    discrete_states
    continuous_outputs
    discrete_outputs
    continuous_random_variables
    discrete_random_variables
    schedules
    models
    resources
    t_next
end
ModelDescription(;
    type = Nothing,
    constants = (;),
    continuous_states = (;),
    discrete_states = (;),
    continuous_outputs = (;),
    discrete_outputs = (;),
    continuous_random_variables = (;),
    discrete_random_variables = (;),
    schedules = (;),
    models = (;),
    resources = (;),
    t_next = NO_T_NEXT,
) = ModelDescription(
    type, constants,
    continuous_states, discrete_states,
    continuous_outputs, discrete_outputs,
    continuous_random_variables, discrete_random_variables,
    schedules, models, resources,
    exact_time(t_next),
)

"""
    validate_model_schedules(description, model_path = "/")

Validate every declared schedule before simulation resources are opened.

Schedules must implement `AbstractSchedule`, and their names must not collide with another
member exposed on the same constructed model. Detecting those mistakes during initialization
produces a focused error instead of relying on named-tuple or model-constructor behavior.
"""
function validate_model_schedules(description::ModelDescription, model_path = "/")

    schedule_names = fieldnames(typeof(description.schedules))
    other_model_member_names = (
        fieldnames(typeof(description.constants))...,
        fieldnames(typeof(description.continuous_states))...,
        fieldnames(typeof(description.discrete_states))...,
        fieldnames(typeof(description.continuous_random_variables))...,
        fieldnames(typeof(description.discrete_random_variables))...,
        fieldnames(typeof(description.models))...,
        fieldnames(typeof(description.resources))...,
    )

    for name in schedule_names

        schedule = strip_fluff_from_variable(description.schedules[name])
        if !(schedule isa AbstractSchedule)
            schedule_path = model_path == "/" ? "/schedules/$name" :
                "$model_path/schedules/$name"
            throw(ArgumentError(
                "Schedule $schedule_path must be an AbstractSchedule, not " *
                "$(typeof(schedule)).",
            ))
        end

        if name in other_model_member_names
            schedule_path = model_path == "/" ? "/schedules/$name" :
                "$model_path/schedules/$name"
            throw(ArgumentError(
                "Schedule $schedule_path conflicts with another model member named $name.",
            ))
        end

    end

    for name in fieldnames(typeof(description.models))
        submodel_path = model_path == "/" ? "/models/$name" : "$model_path/models/$name"
        validate_model_schedules(description.models[name], submodel_path)
    end

    return nothing

end


"""
    collect_schedules!(schedules, description)

Append every schedule in `description` and its submodels to an initialization-time working
vector. The abstract element type is acceptable here because this collection is discarded
after initialization; the runtime scheduler receives a concrete tuple.
"""
function collect_schedules!(
    schedules::Vector{AbstractSchedule},
    description::ModelDescription,
)
    append!(schedules, map(strip_fluff_from_variable, description.schedules))
    for submodel in description.models
        collect_schedules!(schedules, submodel)
    end
    return nothing
end


"""
    collect_unique_schedules(description)

Return one tuple containing the distinct schedules declared throughout a model hierarchy.

Collection and deduplication occur once during initialization. The simulation loop receives
the resulting concrete tuple, allowing dispatch to specialize for built-in and user-defined
schedule types without storing a `Vector{AbstractSchedule}` in the runtime scheduler.
"""
function collect_unique_schedules(description::ModelDescription)
    schedules = AbstractSchedule[]
    collect_schedules!(schedules, description)
    unique!(schedules)
    return Tuple(schedules)
end

"""
Describes a model's continuous-time derivatives and outputs.

* `rates`: A named tuple corresponding with the continuous variables, where each field
  contains the rate of change of that continuous variable.
* `outputs`: A named tuple of continuous-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple contains the `RatesOutput` for each submodel.
* `stop`: Set to true to request that the simulation stop after this accepted sample
  completes. Stop requests from rejected solver attempts and intermediate Runge-Kutta
  stages are ignored.
"""
struct RatesOutput{RT, OT, MT}
    rates::RT
    outputs::OT
    models::MT
    stop::Bool # This could be AbstractStopReason, but that makes this type allocate, which is annoying, so for now, we leave this as bool.
end
RatesOutput(;
    rates = (;),
    outputs = (;),
    models = (;),
    stop = false,
) = RatesOutput(rates, outputs, models, stop)

"""
Describes a model's discrete-time updates and outputs.

* `updates`: A named tuple mapping state name (can be a continuous or discrete state) to the
   updated value
* `outputs`: A named tuple of discrete-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple contains the `UpdatesOutput` for each submodel.
* `t_next`: A replacement for the model's next requested time. When omitted, it defaults to
  `KEEP_T_NEXT` and retains the previous request. `NO_T_NEXT` cancels a finite request.
* `stop`: Set to true to request that the simulation stop after this update is accepted.
"""
struct UpdatesOutput{UT, OT, MT}
    updates::UT
    outputs::OT
    models::MT
    t_next::ExactTime
    stop::Bool
end
UpdatesOutput(;
    updates = (;),
    outputs = (;),
    models = (;),
    t_next = KEEP_T_NEXT,
    stop = false,
) = UpdatesOutput(updates, outputs, models, exact_time(t_next), stop)

"""
    on_triggering(f, schedule::AbstractSchedule, t)

Run `f` and return its result when `schedule` is triggering at official time `t`; otherwise,
return an empty `UpdatesOutput` without evaluating `f`.

The function argument comes first so model update code can use Julia's `do` syntax:

```
function updates_fcn(t, model)
    on_triggering(model.schedule, t) do
        return UpdatesOutput(; updates = (; count = model.count + 1,))
    end
end
```
"""
@inline function on_triggering(f, schedule::AbstractSchedule, t)
    return is_triggering(schedule, t) ? f() : UpdatesOutput()
end

"""
These can be used to decorate the variables in a `ModelDescription`. The decorations become
part of the `TimeSeries` for that variable. Example:

```
VariableDescription(
    SA[1., 2., 3];
    title = "Position",
    dimensions = ["x" => "m", "y" => "m", "z" => "m"],
    interpolator = LinearInterpolation(),
)
```

A `missing` value is allowed, but in that case, the type must be provided explicitly so that
the `TimeSeries` knows what kind of types to expect. Example:

```
VariableDescription{SVector{3, Float64}}(
    missing;
    title = "Position",
    dimensions = ["x" => "m", "y" => "m", "z" => "m"],
)
```

A variable's dimensions can also be grouped together. This only affects plots. Grouped
dimensions will be plotted in a single axis, rather than each dimension getting its own
axis. This can help make plots more compact, and it can be clearer to have multiple lines
sharing a single axis in some cases. By default, each dimension will get its own group.

A variable can also specify the interpolation policy that will be passed through to its
`TimeSeries`. When omitted, the `TimeSeries` chooses its normal default based on whether
the signal is continuous or discrete.
"""
struct VariableDescription{T}
    value::Union{Missing, T}
    title::String
    dimensions::Vector{Dimension}
    groups::Union{Missing, Vector{Pair{String, Vector{String}}}} # Empty/missing groups won't be automatically plotted
    interpolator::Any
    # record::Bool # To let users decide if they want this signal logged (e.g., a weird state or a constant might not be logged).
end
VariableDescription(value; kwargs...) = VariableDescription{typeof(value)}(value; kwargs...)
function VariableDescription{T}(
    value;
    title, dimensions,
    groups = missing, interpolator = missing
) where {T}
    return VariableDescription{T}(
        value, title, Dimension[dimensions...], groups, interpolator,
    )
end

"""
    RandomVariableDescription{T}

Describes a random variable of type `T`. The `f` field should be a function or type
satisfying `f(rng, t)::T` for a discrete random variable or `f(rng, t_last, t_next)::T` for
a continuous random variable. It also stores a `seed::BranchingSeed` for its own random
number generator, which is useful when a model description needs to reproduce the same draw
outside of its usual parent model. The remaining fields, `title`, `dimensions`, and
`groups`, are the same as for `VariableDescription`.
"""
struct RandomVariableDescription{T}
    f::Any
    seed::BranchingSeed
    title::String
    dimensions::Vector{Dimension}
    groups::Union{Missing, Vector{Pair{String, Vector{String}}}} # Empty/missing groups won't be automatically plotted
end
RandomVariableDescription(f; kwargs...) = RandomVariableDescription{Any}(f; kwargs...)
function RandomVariableDescription{T}(
    f;
    seed,
    title,
    dimensions,
    groups = missing,
) where {T}
    return RandomVariableDescription{T}(f, seed, title, Dimension[dimensions...], groups)
end

"""
    RandomVariable{F, T}

Stores a function to take draws from `f::F` using the `rng::Xoshiro` such that `f(rng, t)`
produces a random draw of type `T`, where `t` is time.
"""
struct RandomVariable{F, T}
    f::F
    rng::Xoshiro
end

##################
# User Utilities #
##################

# We don't use these internally; they're helpful modeling tools for users.

"""
    ContinuousWhiteNoise{T}(; sigma::T)

A type that can be used like a function to draw random numbers for a continuous-time process
with the given standard deviation, `sigma::T`. This works for any type that defines
`randn(rng, type)` and broadcasting (Float64, SVector, etc.).

An example:

```
rng = Xoshiro(1)
process = ContinuousWhiteNoise(SA[1., 2.])
process(rng, t_last, t_next) # Yields appropriate random draws.
```
"""
@kwdef struct ContinuousWhiteNoise{T}
    sigma::T
end
function (nu::ContinuousWhiteNoise{T})(rng, t_km1, t_k) where {T}
    return nu.sigma ./ sqrt(t_k - t_km1) .* randn(rng, T)
end

"""
    DiscreteWhiteNoise{T}(; sigma::T)

A type that can be used like a function to draw random numbers for a discrete-time process
with the given standard deviation, `sigma::T`. This works for any type that defines
`randn(rng, type)` and broadcasting (Float64, SVector, etc.).

An example:

```
rng = Xoshiro(1)
process = DiscreteWhiteNoise(SA[1., 2.])
process(rng, t) # Yields appropriate random draws.
```
"""
@kwdef struct DiscreteWhiteNoise{T}
    sigma::T
end
function (nu::DiscreteWhiteNoise{T})(rng, t) where {T}
    return nu.sigma .* randn(rng, T)
end

#########################
# TypedModelDescription #
#########################

"""
This is the same as ModelDescription, except that any VariableDescription stuff has been
pulled out and all types are fixed as type parameters. This is what's used by the sim loop.

This is mutable only to put it on the heap.
"""
@kwdef mutable struct TypedModelDescription{T, CT, XCT, XDT, YCT, YDT, WCT, WDT, ST, MT, RT}
    type::Type{T} # This could actually be any function that takes kwargs.
    constants::CT
    continuous_states::XCT
    discrete_states::XDT
    continuous_outputs::YCT
    discrete_outputs::YDT
    continuous_random_variables::WCT
    discrete_random_variables::WDT
    schedules::ST
    models::MT
    resources::RT
    t_next::ExactTime
end

strip_fluff_from_variable(var) = var
strip_fluff_from_variable(var::VariableDescription) = var.value

# If the user provided something like DiscreteWhiteNoise directly, just use the default
# seed with it.
function strip_fluff_from_random_variable(f::DiscreteWhiteNoise{T}, seed) where {T}
    return RandomVariable{typeof(f), T}(f, Xoshiro(seed))
end
function strip_fluff_from_random_variable(f::ContinuousWhiteNoise{T}, seed) where {T}
    return RandomVariable{typeof(f), T}(f, Xoshiro(seed))
end

# If the user provided a RandomVariable directly, assume they know exactly what they're
# doing and just use it.
strip_fluff_from_random_variable(rv::RandomVariable, seed) = rv

# If the user provided a random variable description, pull out the part we care about.
function strip_fluff_from_random_variable(rvd::RandomVariableDescription{T}, seed) where {T}
    return RandomVariable{typeof(rvd.f), T}(rvd.f, Xoshiro(rvd.seed))
end

# If the user provided something else for taking draws..., well, we don't know what type it
# produces, so we'll have to assume it's an Any. (This should use a
# RandomVariableDescription if they want to be more explicit.)
function strip_fluff_from_random_variable(f, seed)
    return RandomVariable{typeof(f), Any}(f, Xoshiro(seed))
end

function strip_fluff_from_random_variable_set(random_variables, seed)
    return NamedTuple{fieldnames(typeof(random_variables))}(
        map(fieldnames(typeof(random_variables))) do fn
            strip_fluff_from_random_variable(random_variables[fn], seed / string(fn))
        end
    )
end

# This type is used to store resources that need to be closed.
@kwdef struct ResourceManager
    descriptions::Vector{Resources.AbstractResource} = Resources.AbstractResource[]
    payloads::Vector{Any} = Any[]
end
function add_resource!(
    manager::ResourceManager,
    description::Resources.AbstractResource,
    payload,
)
    push!(manager.descriptions, description)
    push!(manager.payloads, payload)
    return nothing
end
function try_to_close_resource(resource, payload)
        try
            Resources.close_resource(resource, payload)
        catch err
            trace = catch_backtrace()
            @error(
                "Failed to close resource = $resource. Continuing...",
                exception = (err, trace),
            )
        end
end
function close_resources(manager::ResourceManager)
    # We close in the reverse order in which we opened. That's probably irrelevant, but it
    # may be useful in some situations.
    for (desc, payload) in Iterators.reverse(zip(manager.descriptions, manager.payloads))
        try_to_close_resource(desc, payload)
    end
    return nothing
end

function create_typed_model_description!(
    manager::ResourceManager,
    desc::ModelDescription, seed::BranchingSeed,
    outdir::Union{Nothing, String}, model_path::String,
)

    # Create the resources one by one, recording a function to close each if something goes
    # wrong.
    payloads = Any[]
    for resource in desc.resources
        payload = Resources.open_resource(resource, ResourceInputs(; outdir, model_path))
        add_resource!(manager, resource, payload)
        push!(payloads, payload)
    end

    # Make the set of resources. We do this first so that our resources are created before
    # our children's -- simply a convention.
    resources = NamedTuple(
        field => payload
        for (field, payload) in zip(fieldnames(typeof(desc.resources)), payloads)
    )

    # Strip the "fluff" from everything, returning just the types we'll need in the loop.
    return TypedModelDescription(;
        type = desc.type,
        constants = map(strip_fluff_from_variable, desc.constants),
        continuous_states = map(strip_fluff_from_variable, desc.continuous_states),
        discrete_states = map(strip_fluff_from_variable, desc.discrete_states),
        continuous_outputs = map(strip_fluff_from_variable, desc.continuous_outputs),
        discrete_outputs = map(strip_fluff_from_variable, desc.discrete_outputs),
        continuous_random_variables = strip_fluff_from_random_variable_set(
            desc.continuous_random_variables, seed,
        ),
        discrete_random_variables = strip_fluff_from_random_variable_set(
            desc.discrete_random_variables, seed,
        ),
        schedules = map(strip_fluff_from_variable, desc.schedules),
        models = NamedTuple(
            field => create_typed_model_description!(
                manager,
                desc.models[field], seed / string(field), outdir,
                model_path * "/" * string(field)
            )
            for field in fieldnames(typeof(desc.models))
        ),
        resources,
        t_next = exact_time(desc.t_next),
    )

end

#########################
# ModelStateDescription #
#########################

# This is our internal representation of the stuff necessary to construct the model form.
# It's mutable only to put it on the heap.
@kwdef mutable struct ModelStateDescription{T, CT, XCT, XDT, WCT, WDT, ST, MT, RT}
    constants::CT
    continuous_states::XCT
    discrete_states::XDT
    continuous_random_variables::WCT
    discrete_random_variables::WDT
    schedules::ST
    models::MT
    resources::RT
    t_next::ExactTime
end
function ModelStateDescription{T}(;
    constants = (;),
    continuous_states = (;),
    discrete_states = (;),
    continuous_random_variables = (;),
    discrete_random_variables = (;),
    schedules = (;),
    models = (;),
    resources = (;),
    t_next = NO_T_NEXT,
) where {T}
    return ModelStateDescription{
        T, typeof(constants), typeof(continuous_states), typeof(discrete_states),
        typeof(continuous_random_variables), typeof(discrete_random_variables),
        typeof(schedules), typeof(models), typeof(resources),
    }(
        constants, continuous_states, discrete_states,
        continuous_random_variables, discrete_random_variables,
        schedules, models, resources,
        exact_time(t_next),
    )
end

# This has no allocations for bits types.
function model(desc::ModelStateDescription{Nothing})
    return (;
        desc.constants...,
        desc.schedules...,
        desc.continuous_states...,
        desc.continuous_random_variables...,
        desc.discrete_states...,
        desc.discrete_random_variables...,
        map(model, desc.models)...,
        desc.resources...,
    )
end

# This has no allocations for bits types.
function model(desc::ModelStateDescription{T}) where {T}
    return T(;
        desc.constants...,
        desc.schedules...,
        desc.continuous_states...,
        desc.continuous_random_variables...,
        desc.discrete_states...,
        desc.discrete_random_variables...,
        map(model, desc.models)...,
        desc.resources...,
    )
end

# This has no allocations for bits types.
function copy_model_state_description_except(
    md::T;
    kwargs...
) where {T <: ModelStateDescription}
    return T(;
        md.constants,
        md.continuous_states,
        md.discrete_states,
        md.continuous_random_variables,
        md.discrete_random_variables,
        md.schedules,
        md.models,
        md.resources,
        md.t_next,
        kwargs...
    )
end

################
# Stop Reasons #
################

"""
The common supertype for every reason a simulation ceased running.

Normal stop requests and failures are deliberately separate categories. A model or hook
request is part of the modeled lifecycle; a numerical or software failure means that
lifecycle could not produce another valid sample.
"""
abstract type AbstractTerminationReason end

"""
A normal, successfully processed request to stop a simulation.
"""
abstract type AbstractStopReason <: AbstractTerminationReason end

"""
A condition that prevented the simulation from producing another valid accepted sample.
"""
abstract type AbstractFailureReason <: AbstractTerminationReason end

"""
Internal sentinel indicating that the simulation loop should continue.
"""
struct UnknownStopReason <: AbstractStopReason end

"""
The simulation successfully processed its requested final sample.
"""
struct ReachedEndTime <: AbstractStopReason
    t_end::ExactTime
end

"""
The first model encountered in deterministic hierarchy order requested a normal stop.
"""
struct ModelRequestedStop <: AbstractStopReason
    model_path::String
    reason::String
end

"""
The first hook encountered in configured order requested a normal stop.
"""
struct HookRequestedStop <: AbstractStopReason
    t::ExactTime
    hook::Hooks.AbstractHook
end

"""
User model code or simulation infrastructure raised an unexpected exception.
"""
struct EncounteredError <: AbstractFailureReason
    time::Float64
    exception::Exception
    trace::Any
end

describe(reason::AbstractTerminationReason) =
    string(typeof(reason))
describe(stop::UnknownStopReason) =
    "The sim stopped for an unknown reason."
describe(stop::ReachedEndTime) =
    "The sim reached the specified end time of $(float(stop.t_end))."
describe(stop::ModelRequestedStop) =
    "A model ($(stop.model_path)) requested a stop: $(stop.reason)."
describe(stop::HookRequestedStop) =
    "A $(stop.hook) hook requested a stop at t = $(float(stop.t))."
describe(stop::EncounteredError) =
    "The sim experienced an error."

##############
# SimOptions #
##############

# We define this generic function before loading the continuous-problem adapter. Its
# concrete hierarchical method remains with the simulation's random-variable machinery,
# below.
function draw_wc end

include("Logs.jl")
include("ContinuousProblems.jl")
include("Solvers.jl")

"""
A set of options for the `simulate` function, with keyword arguments for:

* `outdir`: A directory to save any outputs to (such as `Resources.OutputFile`)
* `log`: Log options to use (e.g., `Logs.BasicLogOptions()`)
* `solver`: Solver to use (e.g., `Solvers.DormandPrince54Options()`)
* `hooks`: A vector of hooks (e.g., `[Hooks.ProgressBarOptions(),]`)
* `time_dimension`: A `Dimension` for the time unit (e.g., `["time" => "s"]`).
"""
@kwdef struct SimOptions
    outdir::Union{Nothing, String} = nothing
    log::Union{Nothing, Logs.AbstractLogOptions} = Logs.BasicLogOptions()
    solver::Solvers.AbstractSolverOptions = Solvers.DormandPrince54Options()
    hooks::Vector{Hooks.AbstractHookOptions} = []
    time_dimension::Dimension = Dimension("time", "s")
    # catch_errors::Bool = true
end

##############
# SimHistory #
##############

"""
A type to store the results from simulation, including fields for:

* `model`: The final model constructed in the sim
* `log`: The log containing the time series for each variable of each model
* `stop`: The normal stop or failure reason that ended the simulation

This type acts like a log itself, so for instance these do the same thing:

```
history["/models/plant"]["position"]
history.log["/models/plant"]["position"]
```

The `keys`, `values`, and `pairs` functions also pass through to the underlying log.
"""
struct SimHistory
    model::ModelDescription
    log::Logs.AbstractLog
    stop::AbstractTerminationReason
end

function Base.show(io::IO, mime::MIME"text/plain", history::SimHistory)
    println(io, "Simulation History:")
    println(io, "  Stop Reason: " * describe(history.stop))
    println(io, "  Model Histories:")
    slugs = sort(collect(keys(history)))
    for slug in slugs
        println(io, "    " * slug)
    end
end

Base.getindex(history::SimHistory, k) = history.log[k]
Base.keys(history::SimHistory) = keys(history.log)
Base.values(history::SimHistory) = values(history.log)
Base.pairs(history::SimHistory) = pairs(history.log)
# TODO: There's more stuff we could pass through.

# We could allow a user to "close" a history, just passing along the call to the log,
# so that they don't have to worry about the internal log, but I'm not sure what the point
# is. The history has fields for a reason. The log is the only thing that needs to be
# closed, and it's reasonable to ask for that directly.
# Logs.close_log(history::SimHistory) = Logs.close_log(history.log)

###########
# Logging #
###########

# These are all of our recursive logging functions.

# A group is active when its model may contribute either kind of logged value at the
# current opportunity. Snapshot state logging is a narrower condition used by the separate
# discrete snapshot traversal below.
@inline function sampling_group_logs_sample(group)
    return group.log_states || group.log_outputs
end

@inline function sampling_groups_log_sample(groups)
    return any(sampling_group_logs_sample, groups)
end

@inline sampling_group_snapshots_states(group) = group.snapshot_states

@inline function sampling_groups_snapshot_states(groups)
    return any(sampling_group_snapshots_states, groups)
end

function update_sampling_groups!(t, groups)

    # The tuple contains each distinct group exactly once, even when many model histories
    # share it. Updating here therefore evaluates each sampler once before tree traversal.
    for group in groups
        Logs.update_sampling_group!(t, group)
    end
    return nothing

end

function log_continuous_stuff!(
    ::ExactTime, ::Float64,
    ::Nothing,
    ::ModelStateDescription,
    ::RatesOutput,
)
end

function log_continuous_states!(t_f, mh_xc, msd_xc)
    for fn in fieldnames(typeof(mh_xc))
        push!(mh_xc[fn], t_f, msd_xc[fn])
    end
end
function log_continuous_outputs!(t_f, mh_yc, ro_yc)
    for fn in fieldnames(typeof(mh_yc))
        if hasfield(typeof(ro_yc), fn)
            push!(mh_yc[fn], t_f, ro_yc[fn])
        end
    end
end

function log_continuous_model! end

# Generate direct field accesses for the heterogeneous child histories. A runtime Symbol
# index makes Julia box those values before the recursive call. The generated code only
# handles this static routing; the logging behavior remains in log_continuous_model!.
@generated function log_continuous_models!(
    t,
    t_f,
    mh_models::MHT,
    msd_models::MSDT,
    ro_models::ROT,
) where {
    MHT <: NamedTuple,
    MSDT <: NamedTuple,
    ROT <: NamedTuple,
}

    statements = map(fieldnames(MHT)) do fn
        field = QuoteNode(fn)
        rates_output = if hasfield(ROT, fn)
            :(getfield(ro_models, $field))
        else
            :(RatesOutput())
        end
        return quote
            model_history = getfield(mh_models, $field)
            if sampling_groups_log_sample(model_history.sampling_groups_in_subtree)
                log_continuous_model!(
                    t, t_f,
                    model_history, getfield(msd_models, $field), $rates_output,
                )
            end
        end
    end
    return quote
        $(statements...)
        nothing
    end

end

function log_continuous_model!(
    t::ExactTime, t_f::Float64,
    mh::Logs.ModelHistory,
    msd::ModelStateDescription,
    ro::RatesOutput,
)

    # The compiled group contains this model's independently evaluated sampling decision.
    sampling_group = mh.sampling_group

    if sampling_group.log_states
        log_continuous_states!(t_f, mh.continuous_states, msd.continuous_states)
    end
    if sampling_group.log_outputs
        log_continuous_outputs!(t_f, mh.continuous_outputs, ro.outputs)
    end
    # TODO: Log the derivatives too.
    log_continuous_models!(t, t_f, mh.models, msd.models, ro.models)

end

function log_continuous_stuff!(
    t::ExactTime, t_f::Float64,
    mh::Logs.ModelHistory,
    msd::ModelStateDescription,
    ro::RatesOutput,
)

    # Evaluate each distinct sampler once, then reject the whole tree when every group is
    # inactive at this accepted time.
    sampling_groups = mh.sampling_groups_in_subtree
    update_sampling_groups!(t, sampling_groups)
    if sampling_groups_log_sample(sampling_groups)
        log_continuous_model!(t, t_f, mh, msd, ro)
    end

end

# Initial values have neither an update event nor a prior state. Record them once directly
# from the typed model description, while still honoring the sampler's initial decision.

function log_initial_discrete_stuff!(t, mh::Nothing, md::TypedModelDescription)
end

function log_initial_discrete_model!(
    t,
    mh::Logs.ModelHistory,
    md::TypedModelDescription,
)

    sampling_group = mh.sampling_group
    if sampling_group.log_states
        for fn in keys(mh.discrete_states)
            push!(mh.discrete_states[fn], float(t), md.discrete_states[fn])
        end
    end
    if sampling_group.log_outputs
        for fn in keys(mh.discrete_outputs)
            push!(mh.discrete_outputs[fn], float(t), md.discrete_outputs[fn])
        end
    end

    for fn in keys(mh.models)
        model_history = mh.models[fn]
        if sampling_groups_log_sample(model_history.sampling_groups_in_subtree)
            log_initial_discrete_model!(t, model_history, md.models[fn])
        end
    end

end

function log_initial_discrete_stuff!(
    t,
    mh::Logs.ModelHistory,
    md::TypedModelDescription,
)

    sampling_groups = mh.sampling_groups_in_subtree
    update_sampling_groups!(t, sampling_groups)
    if sampling_groups_log_sample(sampling_groups)
        log_initial_discrete_model!(t, mh, md)
    end

end

# Discrete logging has two distinct sources:
#
# * Events come from UpdatesOutput. CompleteSampler records sparse state changes and
#   discrete outputs this way.
# * Snapshots come from the post-update ModelStateDescription. RegularSampler records every
#   selected discrete state this way, whether or not it changed in the current update.

function log_discrete_stuff!(
    ::ExactTime, ::Float64,
    ::Nothing,
    ::UpdatesOutput,
    ::ModelStateDescription,
    ::ModelStateDescription,
    include_updated_continuous_states::Bool
)
end

function log_discrete_state_changes!(t_f, mh_xd, uo_updates)
    for fn in fieldnames(typeof(mh_xd))
        if hasfield(typeof(uo_updates), fn)
            push!(mh_xd[fn], t_f, uo_updates[fn])
        end
    end
end

function log_discrete_state_snapshot!(t_f, mh_xd, updated_xd)
    for fn in fieldnames(typeof(mh_xd))
        push!(mh_xd[fn], t_f, updated_xd[fn])
    end
end

function log_continuous_state_updates!(
    t_f, mh_xc, uo_updates, prior_xc,
    include_updated_continuous_states,
)
    for fn in fieldnames(typeof(mh_xc))
        if hasfield(typeof(uo_updates), fn)
            push!(mh_xc[fn], t_f, prior_xc[fn])
            if include_updated_continuous_states
                push!(mh_xc[fn], t_f, uo_updates[fn])
            end
        end
    end
end
function log_discrete_outputs!(t_f, mh_yd, uo_outputs)
    for fn in fieldnames(typeof(mh_yd))
        if hasfield(typeof(uo_outputs), fn)
            push!(mh_yd[fn], t_f, uo_outputs[fn])
        end
    end
end

function log_discrete_event_model! end

# As in continuous logging, generate only the direct field routing needed to preserve each
# heterogeneous child type. Event traversal follows only the models present in the current
# UpdatesOutput; missing branches contain no state changes or outputs to record.
@generated function log_discrete_event_models!(
    t_f,
    mh_models::MHT,
    uo_models::UOT,
    prior_models::PT,
    include_updated_continuous_states,
) where {
    MHT <: NamedTuple,
    UOT <: NamedTuple,
    PT <: NamedTuple,
}

    statements = map(fieldnames(MHT)) do fn
        field = QuoteNode(fn)
        if hasfield(UOT, fn)
            return quote
                model_history = getfield(mh_models, $field)
                if sampling_groups_log_sample(model_history.sampling_groups_in_subtree)
                    log_discrete_event_model!(
                        t_f,
                        model_history, getfield(uo_models, $field),
                        getfield(prior_models, $field),
                        include_updated_continuous_states,
                    )
                end
            end
        else
            return nothing
        end
    end
    return quote
        $(statements...)
        nothing
    end

end

# This is called recursively for the current update event tree.
function log_discrete_event_model!(
    t_f::Float64,
    mh::Logs.ModelHistory,
    uo::UpdatesOutput,
    prior::ModelStateDescription,
    include_updated_continuous_states::Bool,
)

    # The compiled group contains this model's independently evaluated sampling decision.
    sampling_group = mh.sampling_group

    # A discrete state has exactly one owner at this opportunity. Sparse samplers record its
    # UpdatesOutput change here. Snapshot samplers deliberately skip it here because the
    # later snapshot pass records its post-update value; doing both would duplicate the
    # timestamp and value.
    if sampling_group.log_states
        if !sampling_group.snapshot_states
            log_discrete_state_changes!(t_f, mh.discrete_states, uo.updates)
        end

        # Continuous state changes are discontinuity events rather than discrete snapshots.
        # Record the *prior* value at `t`; log_continuous_stuff! records the updated value at
        # the beginning of the next step, which also starts at `t`. At the terminal sample,
        # include_updated_continuous_states requests the right-hand value immediately.
        log_continuous_state_updates!(
            t_f, mh.continuous_states, uo.updates, prior.continuous_states,
            include_updated_continuous_states,
        )
    end

    # Log whatever outputs they provided this time.
    if sampling_group.log_outputs
        log_discrete_outputs!(t_f, mh.discrete_outputs, uo.outputs)
    end

    log_discrete_event_models!(
        t_f, mh.models, uo.models, prior.models,
        include_updated_continuous_states,
    )

end

function log_discrete_snapshot_model! end

# Snapshot traversal follows the fixed model-state hierarchy rather than the sparse update
# tree. Direct field routing keeps independently sampled descendants type-stable.
@generated function log_discrete_snapshot_models!(
    t_f,
    mh_models::MHT,
    updated_models::UMT,
) where {
    MHT <: NamedTuple,
    UMT <: NamedTuple,
}

    statements = map(fieldnames(MHT)) do fn
        field = QuoteNode(fn)
        return quote
            model_history = getfield(mh_models, $field)
            if sampling_groups_snapshot_states(
                model_history.sampling_groups_in_subtree,
            )
                log_discrete_snapshot_model!(
                    t_f,
                    model_history, getfield(updated_models, $field),
                )
            end
        end
    end
    return quote
        $(statements...)
        nothing
    end

end

function log_discrete_snapshot_model!(
    t_f::Float64,
    mh::Logs.ModelHistory,
    updated::ModelStateDescription,
)

    # Only this model's group decides whether its states are recorded. The generated child
    # traversal independently follows branches containing another active snapshot group.
    if mh.sampling_group.snapshot_states
        log_discrete_state_snapshot!(
            t_f, mh.discrete_states, updated.discrete_states,
        )
    end
    log_discrete_snapshot_models!(t_f, mh.models, updated.models)

end

# This is the top-level entry point called right after updating.
function log_discrete_stuff!(
    t::ExactTime, t_f::Float64,
    mh::Logs.ModelHistory,
    uo::UpdatesOutput,
    prior::ModelStateDescription,
    updated::ModelStateDescription,
    include_updated_continuous_states::Bool,
)

    sampling_groups = mh.sampling_groups_in_subtree
    update_sampling_groups!(t, sampling_groups)

    # Event logging sees the sparse update result and pre-update continuous states. Snapshot
    # groups still participate because their discrete outputs and continuous-state
    # discontinuities remain events.
    if sampling_groups_log_sample(sampling_groups)
        log_discrete_event_model!(
            t_f, mh, uo, prior, include_updated_continuous_states,
        )
    end

    # Snapshot logging sees the authoritative post-update state and does not inspect the
    # sparse UpdatesOutput at all.
    if sampling_groups_snapshot_states(sampling_groups)
        log_discrete_snapshot_model!(t_f, mh, updated)
    end

end

#########
# Draws #
#########

# Functions for drawing from the sets of random variables
function draw_crvs(crvs, t_last, t_next)
    return map(crvs) do rv
        return rv.f(rv.rng, t_last, t_next)
    end
end
function draw_drvs(drvs, t)
    return map(drvs) do rv
        return rv.f(rv.rng, t)
    end
end

# We turn off inlining here. This appears to help keep this allocation-free.
@noinline function draw_wc(t_last, t_next, ommd::TypedModelDescription, msd::ModelStateDescription)
    return copy_model_state_description_except(msd;
        continuous_random_variables = draw_crvs(
            ommd.continuous_random_variables, t_last, t_next,
        ),
        models = map(ommd.models, msd.models) do ommd_submodel, msd_submodel
            draw_wc(t_last, t_next, ommd_submodel, msd_submodel)
        end,
    )
end

# We turn off inlining here. This appears to help keep this allocation-free.
@noinline function draw_wd(t, ommd::TypedModelDescription, msd::ModelStateDescription)
    return copy_model_state_description_except(msd;
        discrete_random_variables = draw_drvs(ommd.discrete_random_variables, t),
        models = map(ommd.models, msd.models) do ommd_submodel, msd_submodel
            draw_wd(t, ommd_submodel, msd_submodel)
        end,
    )
end

#########
# Steps #
#########

# We haven't pulled out allocations here since this only happens once, but we could.
function create_model_state(t, ommd::TypedModelDescription{T}) where {T}
    return ModelStateDescription{T}(;
        ommd.constants,
        ommd.continuous_states,
        ommd.discrete_states,
        continuous_random_variables = draw_crvs(
            ommd.continuous_random_variables, float(t), float(t) + 1., # Placeholder value
        ),
        discrete_random_variables = draw_drvs(ommd.discrete_random_variables, t),
        ommd.schedules,
        models = NamedTuple(
            mn => create_model_state(t, ommd.models[mn])
            for mn in keys(ommd.models)
        ),
        ommd.resources,
        ommd.t_next,
    )
end

function update_states(prior_states::T1, updated_states::T2) where {T1, T2}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(T2, f)
                updated_states[f]
            else
                prior_states[f]
            end
        end
    )
end

# Note: the return type parameter here helps this to not allocate, but it might be overly
# restrictive. If types can change, should MSD know about that ahead of time?
#
# `submodels` is a named tuple of MSDs.
# `submodels_updates` is a named tuple (same fields) of UpdatesOutput.
#
function update_submodels(submodels::T1, submodels_updates::T2)::T1 where {T1, T2}

    # A model's `models` section of the UpdatesOutput need not be complete. E.g., if it has
    # a continuous-only model as a submodel, there's no point in "updating" it (a discrete
    # operation). However, in order to make this operation efficient, we'll build a
    # "complete" set of updates, where every model is listed, and if it wasn't in the
    # original submodels_updates, then it will be given an empty UpdatesOutput(). Then,
    # we'll have a named tuple that matches submodels in fields (including their order),
    # and we can just map out `update` function to the corresponding submodels and updates.
    #
    # This is one of our more tedious concessions to efficiency, but honestly, it's not all
    # that bad.
    #
    complete_submodels_updates = NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(T2, f)
                submodels_updates[f]
            else
                UpdatesOutput()
            end
        end
    )

    # Now this map doesn't allocate at all:
    return map(update, submodels, complete_submodels_updates)

end

# Only the dedicated keep instruction preserves the previous request. `NO_T_NEXT` is a
# replacement value that explicitly cancels a finite event.
function update_model_t_next(last_t_next, updated_t_next)
    return updated_t_next == KEEP_T_NEXT ? last_t_next : updated_t_next
end

function update(msd::ModelStateDescription, updates_output::UpdatesOutput)
    return copy_model_state_description_except(
        msd;
        continuous_states = update_states(msd.continuous_states, updates_output.updates),
        discrete_states = update_states(msd.discrete_states, updates_output.updates),
        models = update_submodels(msd.models, updates_output.models),
        t_next = update_model_t_next(msd.t_next, updates_output.t_next),
    )
end

function find_soonest_t_next_from_models(t_last, msd::ModelStateDescription{T}) where {T}
    t_next_from_this_model = if time_isless(t_last, msd.t_next)
        msd.t_next
    else
        NO_T_NEXT # If t_next is in the past, it no longer limits us.
    end
    t_next_from_all_submodels = map(msd.models) do submodel
        find_soonest_t_next_from_models(t_last, submodel)
    end
    t_next_from_this_model = reduce(
        earlier_time, t_next_from_all_submodels;
        init = t_next_from_this_model,
    )
    return t_next_from_this_model
end

"""
    find_model_requested_stop(output)

Return the first model stop request in a deterministic, depth-first traversal of a
`RatesOutput` or `UpdatesOutput` hierarchy.

The parent model is considered before its children, and children follow their named-tuple
field order. For now, one stop reason is retained; this traversal makes "first"
predictable until the public model interface can carry structured reasons and the simulation
history can represent several simultaneous requests.
"""
function find_model_requested_stop(output)

    if output.stop
        return ModelRequestedStop("/", "The model requested that the simulation stop")
    end

    # TODO: This loop allocates, and that seems unnecessary.
    for field in fieldnames(typeof(output.models))

        stop = find_model_requested_stop(output.models[field])
        if !isnothing(stop)

            # We build the model_path in reverse. This prevents the need for us to build a
            # model path for every model, when we only care about the model path once, when
            # we stop.
            if stop isa ModelRequestedStop
                child_path = stop.model_path == "/" ? "" : stop.model_path
                return ModelRequestedStop(
                    "/models/$field$child_path",
                    stop.reason,
                )
            else
                return stop
            end

        end

    end

    return nothing

end

# Preserve the first stop reason encountered while allowing the rest of an accepted sample
# to complete. In particular, hooks and the discrete update still run after an accepted
# beginning-of-step rates evaluation requests a stop.
first_stop(current, candidate) = isnothing(current) ? candidate : current

"""
    step!(...)

Process one complete accepted hybrid-system sample.

The integrator first advances continuous state by exactly one accepted numerical step. The
function then logs its authoritative beginning sample, runs hooks, draws discrete random
variables, and accepts the discrete update at the rational endpoint. Returning establishes
that endpoint as the simulation loop's latest committed time and state. Terminal rate
sampling occurs afterward in `loop!`, where an exception cannot hide the committed sample.
"""
function step!(mh, t, schedules, ommd, problem, updates_fcn, t_last, msd, integrator, hooks)

    # Determine the hard upper bound for one accepted numerical step. Step-size suggestions
    # belong to the runtime integrator; the scheduler owns only exact external boundaries.

    # Assume the next stop is the next time a user asked for a stop (which might be the end
    # time).
    # `searchsortedlast` returns the last requested index whose time is no later than the
    # current official time.
    k_last_requested_stop = searchsortedlast(t, t_last; lt = time_isless)
    if k_last_requested_stop < firstindex(t)
        t_next_from_user = first(t) # This shouldn't really be possible at this point.
    else
        k_next_requested_stop = k_last_requested_stop + 1
        if k_next_requested_stop > lastindex(t)
            t_next_from_user = last(t)
        else
            t_next_from_user = t[k_next_requested_stop]
        end
    end

    # Ask all of the models what time they want to stop next, and take the soonest.
    t_next_from_models = find_soonest_t_next_from_models(t_last, msd)

    # Declarative schedules are global immutable metadata collected during initialization.
    # Their next occurrence is another hard event boundary alongside dynamic model requests.
    t_next_from_schedules = Schedules.find_soonest_time(schedules, t_last)

    t_bound = earlier_time(
        t_next_from_user,
        earlier_time(t_next_from_models, t_next_from_schedules),
    )

    # Ask the integrator for one accepted numerical step. A failure has no accepted
    # endpoint, so hooks and discrete updates must not run for it.
    result = Solvers.step!(
        integrator,
        problem,
        Solvers.StepRequest(t_last, t_bound, msd),
    )
    if result isa Solvers.SolverFailure
        return (t_last, msd, result.reason)
    end

    t_next = result.t_end
    msd = result.state_at_end

    # The beginning state and rates come from the accepted attempt. Rejected attempts and
    # intermediate Runge-Kutta stages never reach logging or model stop handling.
    log_continuous_stuff!(
        t_last, float(t_last),
        mh,
        result.state_at_start,
        result.rates_at_start,
    )
    stop = find_model_requested_stop(result.rates_at_start)

    # Update the hooks.
    #
    # We do this here so that hooks can interact with sim time in a reasonable way. Consider
    # a real-time hook. It will have been initialized at t = 0, at which point it will start
    # a stopwatch. It doesn't want to run the t = 0.1s update until 0.1s have passed since
    # it ran its t = 0 update. By putting this here, we've identified `t_next` and solved
    # the continuous-time stuff up to `t_next`. Now, we run this here, allowing the hook
    # to sleep until its time for the discrete step at t_next to happen.
    #
    if !isempty(hooks)
        m = model(msd)
        for hook in hooks
            hook_outputs = Hooks.update_hook!(hook, t_next, m)
            if hook_outputs.stop
                stop = first_stop(stop, HookRequestedStop(t_next, hook))
            end
        end
    end

    # Make the discrete draws.
    msd = draw_wd(t_next, ommd, msd)

    # Perform the discrete update from t_next^- to t_next^+.
    updates = updates_fcn(t_next, model(msd))

    # A model's first update stop is considered only if an earlier accepted rates evaluation
    # or hook has not already supplied the reason for this sample to be the last.
    stop = first_stop(stop, find_model_requested_stop(updates))

    # Construct the authoritative post-update model state before logging. Sparse event
    # logging still receives `msd` so it can record the pre-update side of continuous-state
    # discontinuities, while snapshot logging reads complete discrete states from
    # `updated_msd`.
    updated_msd = update(msd, updates)

    # Log the update events and any state snapshots selected at this time.
    #
    # If a discrete update changes continuous state, this records the pre-update side of the
    # discontinuity. The next accepted rates sample, or the explicit terminal sample below,
    # records the post-update side together with matching continuous outputs.
    #
    log_discrete_stuff!(
        t_next, float(t_next), mh,
        updates, msd, updated_msd, false,
    )

    # Now accept the update.
    msd = updated_msd

    return (t_next, msd, isnothing(stop) ? UnknownStopReason() : stop)

end

########################
# Model Initialization #
########################

get_branching_seed(seed::BranchingSeed) = seed
get_branching_seed(seed::Integer) = BranchingSeed(seed, "")

# Collect and normalize the settings shared by every initialization interface.
function initialization_context(;
    t_start = 0//1,
    model_path = "",
    seed = BranchingSeed(0, model_path),
    outdir = nothing,
)
    return (; t_start, model_path, seed = get_branching_seed(seed), outdir)
end

# Run the user's initialization function before constructing the simulation artifacts.
function create_initialization_artifacts(init_fcn, user_data, context)

    model_description = init_fcn(context.t_start, user_data, context.seed)
    return create_initialization_artifacts(model_description, context)

end

# Construct the typed model description, model state, schedules, and resource manager from
# a model description and normalized initialization context.
function create_initialization_artifacts(
    model_description::ModelDescription,
    context,
)

    # Schedules are immutable declarations, so validate and collect them before opening any
    # model resources. The concrete, deduplicated tuple becomes simulation-level scheduler
    # metadata, while each named declaration also remains available on its model form.
    validate_model_schedules(model_description)
    schedules = collect_unique_schedules(model_description)

    # We'll keep track of resources we create along the way so that we can close them.
    manager = ResourceManager()

    try

        # We should be done with VariableDescriptions, etc., at this point. Now, we can
        # strip all of those out to obtain the simplified TypedModelDescription.
        ommd = create_typed_model_description!(
            manager,
            model_description,
            context.seed,
            context.outdir,
            context.model_path,
        )

        # We can now fill in the draws to have a complete "model state description", from
        # which we can construct the model form.
        msd = create_model_state(context.t_start, ommd)

        return (; model_description, ommd, msd, schedules, manager)

    catch

        # Make sure we close anything we opened along the way.
        close_resources(manager)

        # Now return to the regular error.
        rethrow()

    end

end

"""
    Base.close(desc::ModelDescription, m)

Closes all resources described in the `desc` ModelDescription, where `m` is an instance of
the model.
"""
function Base.close(desc::ModelDescription, m)

    # Let the submodels close their resources, in the reverse order in which we opened them.
    for mn in Iterators.reverse(fieldnames(typeof(desc.models)))
        Base.close(desc.models[mn], getproperty(m, mn))
    end

    # Close this model's resources, again in reverse order.
    for fn in Iterators.reverse(fieldnames(typeof(desc.resources)))
        try_to_close_resource(desc.resources[fn], getproperty(m, fn))
    end

    return nothing

end

##############
# Simulation #
##############

# Hooks may open resources, so we want to make sure we can always call the close function
# for them.
function close_hooks(hooks, t_end, final_model)
    for hook in Iterators.reverse(hooks)
        try
            Hooks.close_hook!(hook, t_end, final_model)
        catch err
            trace = catch_backtrace()
            @error "Failed to close hook = $hook. Continuing..." exception = (err, trace)
        end
    end
end

# Build all of the things necessary for the loop and subsequent tear-down.
function make_runtime(inputs)

    # This might be a tuple with (t_start, t_end), but it can also be any collection of
    # monotonic times.
    t = [exact_time(el) for el in inputs.t]
    t_start = first(t)

    # Pull out the full model description from the initialization function, as well as the
    # typed model description, and finally the model state description.
    context = initialization_context(;
        t_start,
        seed = inputs.seed,
        outdir = inputs.options.outdir,
    )
    artifacts = create_initialization_artifacts(
        inputs.init_fcn,
        inputs.user_data,
        context,
    )
    (; model_description, ommd, msd, schedules, manager) = artifacts

    # Now that resources are open, we need to make absolutely sure we close them.
    try

        # Use those descriptions to set up the time histories.
        log, mh = Logs.create_log(
            inputs.options.log, model_description, inputs.options.time_dimension,
        )

        # Log the initial stuff.
        log_initial_discrete_stuff!(t_start, mh, ommd)

        # Adapt the hierarchical model to the small mathematical interface consumed by
        # continuous-time integrators, then create runtime solver state for this simulation.
        problem = ContinuousProblems.ContinuousProblem(ommd, inputs.rates_fcn)
        integrator = Solvers.create_integrator(inputs.options.solver, problem, msd)

        # Create the hooks one at a time. If any throws, close the already-opened hooks and
        # then rethrow.
        initial_model = model(msd)
        hooks = Hooks.AbstractHook[]
        try
            for hook_options in inputs.options.hooks
                push!(hooks, Hooks.create_hook(hook_options, t, initial_model))
            end
        catch err
            close_hooks(hooks, t_start, initial_model)
            rethrow(err)
        end

        return (;
            inputs.updates_fcn, inputs.close_fcn,
            model_description, ommd,
            t, schedules,
            msd,
            log, mh,
            problem, integrator,
            hooks, manager,
        )

    catch err

        # Close any open resources.
        close_resources(manager)
        rethrow(err)

    end

end

"""
    loop!(...)

Run accepted hybrid-system samples until a normal stop or failure is reported.

This function is the exception boundary for simulation execution. Numerical failures arrive
as ordinary solver results; unexpected Julia exceptions are captured as `EncounteredError`
so resources, hooks, and logs can still be closed by `simulate`.
"""
function loop!(runtime)

    # Pull these out here so we aren't constantly pulling them in the loop.
    mh = runtime.mh
    t = runtime.t
    schedules = runtime.schedules
    ommd = runtime.ommd
    problem = runtime.problem
    updates_fcn = runtime.updates_fcn
    integrator = runtime.integrator
    hooks = runtime.hooks
    t_end = last(runtime.t)

    # These are updated by the loop.
    t_completed = first(runtime.t)
    msd = runtime.msd
    stop = UnknownStopReason()

    # No matter what happens, this function returns all of the progress it's made.
    try

        while isa(stop, UnknownStopReason)

            # `step!` returns only after the accepted continuous endpoint and its discrete
            # update are complete. Assigning its result here is the simulation's commit
            # point: later failures must retain this time and state.
            t_completed, msd, stop = step!(
                mh, t, schedules, ommd, problem, updates_fcn, t_completed, msd,
                integrator, hooks
            )

            # A successfully processed terminal sample receives one direct post-update
            # rates evaluation. Solver failures did not accept a sample and therefore must
            # not run this path, even if their reported time happens to equal `t_end`.
            should_sample_terminal_rates = (
                stop isa AbstractStopReason &&
                (t_completed == t_end || !(stop isa UnknownStopReason))
            )
            if should_sample_terminal_rates

                # `UnknownStopReason` means that reaching `t_end` initiated termination; it
                # is an internal loop sentinel, not a reason that should take precedence
                # over a terminal model request or `ReachedEndTime`.
                terminal_stop = stop isa UnknownStopReason ? nothing : stop
                terminal_rates = ContinuousProblems.evaluate_rates(
                    problem,
                    float(t_completed),
                    msd,
                )
                log_continuous_stuff!(
                    t_completed, float(t_completed),
                    mh, msd, terminal_rates,
                )
                terminal_stop = first_stop(
                    terminal_stop,
                    find_model_requested_stop(terminal_rates),
                )
                stop = first_stop(
                    terminal_stop,
                    t_completed == t_end ? ReachedEndTime(t_end) : nothing,
                )

            end

        end

    catch err

        trace = catch_backtrace()
        @error "The simulation encounted an error." exception = (err, trace)
        stop = EncounteredError(float(t_completed), err, stacktrace(trace))

    end

    return (; t_completed, msd, stop)

end

# Produces the final model, closes the open resources, and wraps up the results.
function tear_down(runtime, loop_outputs)
    final_model = nothing
    try
        final_model = model(loop_outputs.msd)
        runtime.close_fcn(loop_outputs.t_completed, final_model)
        return (;
            history = SimHistory(runtime.model_description, runtime.log, loop_outputs.stop),
            t_final = loop_outputs.t_completed,
            final_model,
        )
    finally
        close_hooks(runtime.hooks, loop_outputs.t_completed, final_model)
        close_resources(runtime.manager)
    end
end

#############
# Interface #
#############

"""
    initialize(user_data; init_fcn, seed = 0, t_start = 0)

Creates the initial model form from the given `user_data`, `init_fcn`, `seed`, and start
time, `t_start`. See `simulate` for a definition of these inputs.

If the `model_description` contains any resources (open files, connections), call
`Base.close(model_description, model)` to release those resources. Alternatively, consider
using the `do` form: `initialize(user_data) do model ... end`.
"""
function initialize(user_data; init_fcn, kwargs...)
    context = initialization_context(; kwargs...)
    artifacts = create_initialization_artifacts(init_fcn, user_data, context)
    return model(artifacts.msd)
end

"""
    initialize(model_description; seed = 0, t_start = 0)

Creates the initial model form from the given `ModelDescription`, `seed`, and start time,
`t_start`.

If the `model_description` contains any resources (open files, connections), call
`Base.close(model_description, model)` to release those resources. Alternatively, consider
using the `do` form: `initialize(model_description) do model ... end`.
"""
function initialize(model_description::ModelDescription; kwargs...)
    context = initialization_context(; kwargs...)
    artifacts = create_initialization_artifacts(model_description, context)
    return model(artifacts.msd)
end

# Run a function with an initialized model and close its resources afterward.
function use_initialized_model(f, artifacts)
    try
        return f(model(artifacts.msd))
    finally
        close_resources(artifacts.manager)
    end
end

"""
    initialize(f, user_data; init_fcn, kwargs...)

This form of `initialize` allows the `do` pattern, like so:

```
initialize(user_data; init_fcn = ..., kwargs...) do m
    # Do something with model m.
    ...
end
```

This is useful when a model opens a resource, like a file. When the `do` block is finished,
this function will automatically close all opened resources, even if there was an error.

Optional keyword arguments:

* `seed`: An integer or `BranchingSeed`
* `t_start`: The time to use for initialization
"""
function initialize(f::Function, user_data; init_fcn, kwargs...)
    context = initialization_context(; kwargs...)
    artifacts = create_initialization_artifacts(init_fcn, user_data, context)
    return use_initialized_model(f, artifacts)
end

"""
    initialize(f, model_description::ModelDescription; kwargs...)

This form of `initialize` allows the `do` pattern:

```
initialize(model_description; kwargs...) do m
    # Do something with model m.
    ...
end
```

This is useful when a model opens a resource, like a file. When the `do` block is finished,
this function will automatically close all opened resources, even if there was an error.

Optional keyword arguments:

* `seed`: An integer or `BranchingSeed`
* `t_start`: The time to use for initialization
"""
function initialize(f::Function, model_description::ModelDescription; kwargs...)
    context = initialization_context(; kwargs...)
    artifacts = create_initialization_artifacts(model_description, context)
    return use_initialized_model(f, artifacts)
end

"""
    simulate(user_data; t, init_fcn, rates_fcn, updates_fcn, close_fcn, seed, options)

Runs a simulation, returning the time history, end time, and final model.

* `user_data`: Can be anything used by the `init_fcn`
* `t`: A collection of monotonic times. The sim will step to exactly each given time, plus
  as many other steps are required by the solver and models. At the very least, this must
  contain a start time and end time.
* `init_fcn`: Will be called with `(t_start, user_data, seed)`, where `t_start` is the first
  element of the above `t` input. This must return a `ModelDescription`.
* `rates_fcn`: Will be called with `(t, model)` and is expected to return a `RatesOutput`.
* `updates_fcn`: Will be called with `(t, model)` and is expected to return an
  `UpdatesOutput`.
* `close_fcn`: Will be called when simulation completes (even if an error is caught) with
  `(t, model)`. No return value is expected.
* `seed`: A top-level seed (Int) to control all random number generation in the sim. The
  `init_fcn` receives this as a `BranchingSeed`.
* `options`: See `SimOptions`.
"""
function simulate(
    user_data;
    t,
    init_fcn,
    rates_fcn = (args...) -> RatesOutput(),
    updates_fcn = (args...) -> UpdatesOutput(),
    close_fcn = (t, model) -> nothing,
    seed::Union{Integer, BranchingSeed} = 0,
    options::SimOptions = SimOptions(),
)
    inputs = (; user_data, t, init_fcn, rates_fcn, updates_fcn, close_fcn, seed, options)
    runtime = make_runtime(inputs)
    loop_outputs = loop!(runtime)
    results = tear_down(runtime, loop_outputs)
    return (results.history, results.t_final, results.final_model)
end

end # module SystemsOfSystems
