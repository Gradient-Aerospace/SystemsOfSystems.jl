module SystemsOfSystems

# Running simulations
export initialize, simulate, SimHistory, SimOptions, succeeded,
    Schedules, Solvers, Hooks, Logs, Resources

# Model descriptions
export ModelDescription, VariableDescription, RandomVariableDescription,
    RatesOutput, UpdatesOutput,
    OutputFile, Resource

# Utilities
export Dimension,
    BranchingSeed, branch,
    TimeSeries, AbstractTimeSeriesInterpolator, SampleAndHold, LinearInterpolation, select,
    plot_ts,
    ContinuousWhiteNoise, DiscreteWhiteNoise,
    AbstractSchedule, RegularSchedule, OffsetRegularSchedule,
    on_triggering, is_triggering, next_trigger_time, next_regular_time,
    KEEP_T_NEXT, NO_T_NEXT,
    is_regular_step_triggering, # backward compatibility
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
A description of the model structure returned by the `init_fcn` provided to `simulate`. It
contains:

* `type::Type`: The type that should be used when constructing the model, or `Nothing` to
  use a named tuple. The type should accept keyword arguments for the variables below.
* `constants`: A named tuple of each constant the model should hold
* `continuous_states`: A named tuple of each of the continuous states in the model
* `discrete_states`: A named tuple of each of the discrete states in the model
* `continuous_outputs`: A named tuple of each of the continuous outputs in the model
* `discrete_outputs`: A named tuple of each of the discrete outputs in the model
* `continuous_random_variables`: A named tuple of each of the continuous random variables in
  the model. Each element can be a function mapping `(rng, t_km1, dt_f)` to a value, or a
  `RandomVariableDescription`. The interval starts at the exact time `t_km1` and has the
  floating-point duration `dt_f`.
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

Within each model, names must be unique across constants, states, outputs, random variables,
schedules, submodels, and resources. Initialization throws an `ArgumentError` describing
any conflicts before opening model resources.
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
    validate_model_description(description, model_path = "/")

Validates the names and schedules declared by every model before simulation resources are
opened.

Variable names must be unique across every named category in a model. Schedules must also
implement `AbstractSchedule`. Detecting these mistakes during initialization produces a
focused error instead of relying on named-tuple, history-lookup, or model-constructor
behavior.
"""
function validate_model_description(description::ModelDescription, model_path = "/")

    categories = (;
        constants = fieldnames(typeof(description.constants)),
        continuous_states = fieldnames(typeof(description.continuous_states)),
        discrete_states = fieldnames(typeof(description.discrete_states)),
        continuous_outputs = fieldnames(typeof(description.continuous_outputs)),
        discrete_outputs = fieldnames(typeof(description.discrete_outputs)),
        continuous_random_variables =
            fieldnames(typeof(description.continuous_random_variables)),
        discrete_random_variables =
            fieldnames(typeof(description.discrete_random_variables)),
        schedules = fieldnames(typeof(description.schedules)),
        models = fieldnames(typeof(description.models)),
        resources = fieldnames(typeof(description.resources)),
    )

    all_names = Symbol[]
    for names in categories
        append!(all_names, names)
    end
    for name in unique(all_names)
        conflicting_categories = Symbol[
            category
            for category in keys(categories) if name in categories[category]
        ]
        if length(conflicting_categories) > 1
            category_list = join(
                ("`$category`" for category in conflicting_categories),
                ", ",
            )
            throw(ArgumentError(
                "Model $model_path uses the name `$name` in multiple categories: " *
                "$category_list. Variable names must be unique within each model.",
            ))
        end
    end

    for name in fieldnames(typeof(description.schedules))

        schedule = strip_fluff_from_variable(description.schedules[name])
        if !(schedule isa AbstractSchedule)
            schedule_path = model_path == "/" ? "/schedules/$name" :
                "$model_path/schedules/$name"
            throw(ArgumentError(
                "Schedule $schedule_path must be an AbstractSchedule, not " *
                "$(typeof(schedule)).",
            ))
        end

    end

    for name in fieldnames(typeof(description.models))
        submodel_path = model_path == "/" ? "/models/$name" : "$model_path/models/$name"
        validate_model_description(description.models[name], submodel_path)
    end

    return nothing

end


"""
    collect_schedules!(schedules, description)

Appends every schedule in `description` and its submodels to an initialization-time working
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

Returns one tuple containing the distinct schedules declared throughout a model hierarchy.

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
A container for a model's continuous-time derivatives and outputs.

* `rates`: A named tuple corresponding with the continuous variables, where each field
  contains the rate of change of that continuous variable.
* `outputs`: A named tuple of continuous-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple containing the `RatesOutput` for each submodel.
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
A container for a model's discrete-time updates and outputs. A model that has no updates,
outputs, replacement `t_next`, or stop request at a sample may return `nothing` instead of
an empty `UpdatesOutput()`.

* `updates`: A named tuple mapping state name (can be a continuous or discrete state) to the
   updated value
* `outputs`: A named tuple of discrete-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple containing the `UpdatesOutput` or `nothing` for each submodel.
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

Runs `f` and returns its result when `schedule` is triggering at official time `t`;
otherwise, returns `nothing` without evaluating `f`.

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
    return is_triggering(schedule, t) ? f() : nothing
end

"""
A container for a variable's initial value and optional logging metadata in a
`ModelDescription`. The metadata becomes part of the `TimeSeries` for that variable.
Example:

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

A container for a random variable of type `T`. The `f` field is a function or callable type
satisfying `f(rng, t)::T` for a discrete random variable or
`f(rng, t_km1, dt_f)::T` for a continuous random variable, where `t_km1` is the exact
simulation time at the beginning of the interval and `dt_f` is its floating-point duration.
It also stores a `seed::BranchingSeed` for its own random number generator, which is useful
when a model description needs to reproduce the same draw outside of its usual parent
model. The remaining fields, `title`, `dimensions`, and `groups`, are the same as for
`VariableDescription`.
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

A container for a callable random process, `f::F`, and its `rng::Xoshiro`, where
`f(rng, t)` produces a random draw of type `T` at time `t`.
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

A callable Gaussian white-noise process for continuous-time models with the given standard
deviation, `sigma::T`. This works for any type that defines `randn(rng, type)` and
broadcasting (`Float64`, `SVector`, etc.).

An example:

```
rng = Xoshiro(1)
process = ContinuousWhiteNoise(SA[1., 2.])
process(rng, t_km1, dt_f) # Yields appropriate random draws.
```
"""
@kwdef struct ContinuousWhiteNoise{T}
    sigma::T
end
function (nu::ContinuousWhiteNoise{T})(rng, t_km1, dt_f) where {T}
    dt_f > 0 || throw(ArgumentError("ContinuousWhiteNoise requires a positive duration."))
    return nu.sigma ./ sqrt(dt_f) .* randn(rng, T)
end

"""
    DiscreteWhiteNoise{T}(; sigma::T)

A callable Gaussian white-noise process for discrete-time models with the given standard
deviation, `sigma::T`. This works for any type that defines `randn(rng, type)` and
broadcasting (`Float64`, `SVector`, etc.).

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
An internal model description with the `VariableDescription` metadata removed and all types
fixed as type parameters. This is used by the simulation loop.

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
    models_have_continuous_random_variables::Bool
    models_have_discrete_random_variables::Bool
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

    # Create the typed submodels first so that we can record which random-variable draw
    # processes need to continue into them during the simulation loop.
    models = NamedTuple(
        field => create_typed_model_description!(
            manager,
            desc.models[field], seed / string(field), outdir,
            model_path * "/" * string(field)
        )
        for field in fieldnames(typeof(desc.models))
    )
    models_have_continuous_random_variables = any(models) do model
        return !isempty(model.continuous_random_variables) ||
            model.models_have_continuous_random_variables
    end
    models_have_discrete_random_variables = any(models) do model
        return !isempty(model.discrete_random_variables) ||
            model.models_have_discrete_random_variables
    end

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
        models,
        models_have_continuous_random_variables,
        models_have_discrete_random_variables,
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
include("SimulationLogging.jl")
include("ContinuousProblems.jl")
include("Solvers.jl")

"""
A container for the options supplied to `simulate`, with fields for:

* `outdir`: A directory to save any outputs to (such as `Resources.OutputFile`)
* `log`: Log options to use (e.g., `Logs.BasicLogOptions()`)
* `solver`: Solver to use (e.g., `Solvers.DormandPrince54Options()`)
* `hooks`: A vector of hooks (e.g., `[Hooks.ProgressBarOptions(),]`)
* `time_dimension`: A `Dimension` for the time unit (e.g., `"time" => "s"`).
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
A container for simulation results, including fields for:

* `t_start`: The simulation's start time
* `t_stop`: The last time completed by the simulation
* `log`: The log containing the time series for each variable of each model
* `model`: The final model constructed in the sim
* `stop`: The normal stop or failure reason that ended the simulation

This type acts like a log itself, so for instance these do the same thing:

```
history["/models/plant"]["position"]
history.log["/models/plant"]["position"]
```

The `keys`, `values`, and `pairs` functions also pass through to the underlying log.
"""
struct SimHistory{M}
    t_start::ExactTime
    t_stop::ExactTime
    log::Logs.AbstractLog
    model::M
    stop::AbstractTerminationReason
end

"""
    succeeded(h::SimHistory)

Returns true if the simulation ended without throwing an error or failing to converge on a
solution.
"""
succeeded(h::SimHistory) = !(h.stop isa AbstractFailureReason)

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

#########
# Draws #
#########

# Functions for drawing from the sets of random variables
function draw_crvs(crvs, t_km1, dt_f)
    return map(crvs) do rv
        return rv.f(rv.rng, t_km1, dt_f)
    end
end
function draw_drvs(drvs, t)
    return map(drvs) do rv
        return rv.f(rv.rng, t)
    end
end

# We turn off inlining here. This appears to help keep this allocation-free.
@noinline function draw_wc(
    t_km1,
    dt_f,
    ommd::TypedModelDescription,
    msd::ModelStateDescription,
)

    # Reuse the entire state description when neither this model nor its descendants have
    # any continuous random variables.
    if isempty(ommd.continuous_random_variables) &&
        !ommd.models_have_continuous_random_variables
        return msd
    end

    models = if ommd.models_have_continuous_random_variables
        map(ommd.models, msd.models) do ommd_submodel, msd_submodel
            draw_wc(t_km1, dt_f, ommd_submodel, msd_submodel)
        end
    else
        msd.models
    end

    return copy_model_state_description_except(msd;
        continuous_random_variables = draw_crvs(
            ommd.continuous_random_variables, t_km1, dt_f,
        ),
        models,
    )

end

# We turn off inlining here. This appears to help keep this allocation-free.
@noinline function draw_wd(t, ommd::TypedModelDescription, msd::ModelStateDescription)

    # Reuse the entire state description when neither this model nor its descendants have
    # any discrete random variables.
    if isempty(ommd.discrete_random_variables) &&
        !ommd.models_have_discrete_random_variables
        return msd
    end

    models = if ommd.models_have_discrete_random_variables
        map(ommd.models, msd.models) do ommd_submodel, msd_submodel
            draw_wd(t, ommd_submodel, msd_submodel)
        end
    else
        msd.models
    end

    return copy_model_state_description_except(msd;
        discrete_random_variables = draw_drvs(ommd.discrete_random_variables, t),
        models,
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
            ommd.continuous_random_variables, t, 1., # Placeholder duration
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
# `submodels_updates` is a named tuple of UpdatesOutput or `nothing` values.
#
function update_submodels(submodels::T1, submodels_updates::T2)::T1 where {T1, T2}

    # A model's `models` section of the UpdatesOutput need not be complete. E.g., if it has
    # a continuous-only model as a submodel, there's no point in "updating" it (a discrete
    # operation). However, in order to make this operation efficient, we'll build a
    # "complete" set of updates, where every model is listed, and if it wasn't in the
    # original submodels_updates, then it will be given `nothing`. Then,
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
                nothing
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

update(msd::ModelStateDescription, ::Nothing) = msd

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

function prepend_model_stop_path(stop::ModelRequestedStop, field::Symbol)
    child_path = stop.model_path == "/" ? "" : stop.model_path
    return ModelRequestedStop(
        "/models/$field$child_path",
        stop.reason,
    )
end
prepend_model_stop_path(stop, ::Symbol) = stop

# A generated straight-line traversal preserves named-tuple field order without the
# recursive Base.tail types that become expensive to infer for wide model hierarchies. Each
# emitted block performs the ordinary recursive search for one child and returns early
# when that child contains the first request.
@generated function find_model_requested_stop_in_models(models::M) where {M <: NamedTuple}

    statements = map(fieldnames(M)) do field
        return quote
            stop = find_model_requested_stop(getfield(models, $(QuoteNode(field))))
            if !isnothing(stop)
                return prepend_model_stop_path(stop, $(QuoteNode(field)))
            end
        end
    end
    return Expr(:block, statements..., :(return nothing))

end

"""
    find_model_requested_stop(output)

Returns the first model stop request in a deterministic, depth-first traversal of a
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
    return find_model_requested_stop_in_models(output.models)

end

find_model_requested_stop(::Nothing) = nothing

# Preserve the first stop reason encountered while allowing the rest of an accepted sample
# to complete. In particular, hooks and the discrete update still run after an accepted
# beginning-of-step rates evaluation requests a stop.
first_stop(current, candidate) = isnothing(current) ? candidate : current

"""
    step!(...)

Processes one complete accepted hybrid-system sample.

The function first records the known continuous state, then asks the integrator to advance
it by exactly one accepted numerical step. An accepted result supplies the corresponding
continuous outputs. The function then runs hooks, draws discrete random variables, and
accepts the discrete update at the rational endpoint. Returning establishes that endpoint
as the simulation loop's latest committed time and state. Terminal rate sampling occurs
afterward in `loop!`, where an exception cannot hide the committed sample.
"""
function step!(
    logging_runtime,
    t,
    schedules,
    ommd,
    problem,
    updates_fcn,
    t_last,
    msd,
    integrator,
    hooks,
)

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

    # Record the committed state before entering the fallible solver. The matching output
    # is recorded only if the solver returns authoritative beginning-of-step rates.
    SimulationLogging.log_continuous_state_stuff!(
        t_last, float(t_last), logging_runtime, msd,
    )

    # Ask the integrator for one accepted numerical step. A failure has no accepted
    # endpoint, so outputs, hooks, and discrete updates must not run for it.
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

    # The beginning rates come from the accepted attempt. Rejected attempts and intermediate
    # Runge-Kutta stages never reach output logging or model stop handling.
    SimulationLogging.log_continuous_output_stuff!(
        float(t_last), logging_runtime, result.rates_at_start,
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
    # discontinuity. The next continuous state sample records the post-update side. Its
    # matching output is recorded separately after rates have been evaluated successfully.
    #
    SimulationLogging.log_discrete_stuff!(
        t_next, float(t_next), logging_runtime,
        updates, msd, updated_msd,
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

    # Validate the model description before opening any resources. The concrete,
    # deduplicated schedule tuple then becomes simulation-level scheduler metadata, while
    # each named schedule declaration also remains available on its model form.
    validate_model_description(model_description)
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

function validate_requested_times(t)

    length(t) >= 2 || throw(ArgumentError(
        "t must contain at least a start time and a stop time.",
    ))

    for k in eachindex(t)

        isfinite(t[k]) || throw(ArgumentError("t[$k] must be finite."))
        if k != firstindex(t) && !time_isless(t[k - 1], t[k])
            throw(ArgumentError(
                "t must be strictly increasing, but t[$(k - 1)] = $(t[k - 1]) " *
                "and t[$k] = $(t[k]).",
            ))
        end

    end

    return nothing

end

# Build all of the things necessary for the loop and subsequent tear-down.
function make_runtime(inputs)

    # This might be a tuple with (t_start, t_end), but it can also be any collection of
    # strictly increasing times.
    t = [exact_time(el) for el in inputs.t]
    validate_requested_times(t)
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
        logging_runtime = Logs.create_model_logging_runtime(mh, inputs.options.log)

        # Log the initial stuff.
        SimulationLogging.log_initial_discrete_stuff!(t_start, logging_runtime, ommd)

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
            inputs.updates_fcn,
            ommd,
            t, schedules,
            msd,
            log, logging_runtime,
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

Runs accepted hybrid-system samples until a normal stop or failure is reported.

This function is the exception boundary for simulation execution. Numerical failures arrive
as ordinary solver results; unexpected Julia exceptions are captured as `EncounteredError`
so resources, hooks, and logs can still be closed by `simulate`.
"""
function loop!(runtime)

    # Pull these out here so we aren't constantly pulling them in the loop.
    logging_runtime = runtime.logging_runtime
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
                logging_runtime,
                t, schedules, ommd, problem, updates_fcn, t_completed, msd,
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
                SimulationLogging.log_continuous_state_stuff!(
                    t_completed, float(t_completed), logging_runtime, msd,
                )
                terminal_rates = ContinuousProblems.evaluate_rates(
                    problem,
                    float(t_completed),
                    msd,
                )
                SimulationLogging.log_continuous_output_stuff!(
                    float(t_completed), logging_runtime, terminal_rates,
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
        return SimHistory(
            first(runtime.t),
            loop_outputs.t_completed,
            runtime.log,
            final_model,
            loop_outputs.stop,
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
    simulate(user_data; t, init_fcn, rates_fcn, updates_fcn, seed, options)

Runs a simulation, returning a `SimHistory` containing its log, start and stop times, final
model, and termination reason.

* `user_data`: Can be anything used by the `init_fcn`
* `t`: A collection of strictly increasing, finite times. The sim will step to exactly each
  given time, plus as many other steps as are required by the solver and models. At the very
  least, this must contain a start time and end time.
* `init_fcn`: Will be called with `(t_start, user_data, seed)`, where `t_start` is the first
  element of the above `t` input. This must return a `ModelDescription`.
* `rates_fcn`: Will be called with `(t, model)` and is expected to return a `RatesOutput`.
* `updates_fcn`: Will be called with `(t, model)` and is expected to return an
  `UpdatesOutput`, or `nothing` when there are no updates, outputs, replacement `t_next`, or
  stop request.
* `seed`: A top-level seed (Int) to control all random number generation in the sim. The
  `init_fcn` receives this as a `BranchingSeed`.
* `options`: See `SimOptions`.
"""
function simulate(
    user_data;
    t,
    init_fcn,
    rates_fcn = (args...) -> RatesOutput(),
    updates_fcn = (args...) -> nothing,
    seed::Union{Integer, BranchingSeed} = 0,
    options::SimOptions = SimOptions(),
)
    inputs = (; user_data, t, init_fcn, rates_fcn, updates_fcn, seed, options)
    runtime = make_runtime(inputs)
    loop_outputs = loop!(runtime)
    return tear_down(runtime, loop_outputs)
end

end # module SystemsOfSystems
