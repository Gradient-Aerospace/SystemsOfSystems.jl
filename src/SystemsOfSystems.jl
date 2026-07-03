module SystemsOfSystems

# Running simulations
export initialize, simulate, SimOptions, Solvers, Hooks, Logs

# Modeling
export ModelDescription, VariableDescription, RandomVariableDescription, RatesOutput,
    UpdatesOutput
export is_regular_step_triggering

# Utilities
export BranchingSeed, branch
export Dimension, TimeSeries, AbstractTimeSeriesInterpolator,
    SampleAndHold, LinearInterpolation

using Random: Xoshiro
import Random

include("TimeSeries.jl")
using .TimeSeriesStuff

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
* `models`: A named tuple containing the ModelDescription of each submodel.
* `t_next`: The next sim time at which the model requests that the integrator stop. The
  integrator will step no latter than this time, but may step earlier.

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
    models
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
    models = (;),
    t_next = 0//1,
) = ModelDescription(
    type, constants,
    continuous_states, discrete_states,
    continuous_outputs, discrete_outputs,
    continuous_random_variables, discrete_random_variables,
    models,
    rationalize(t_next),
)

"""
Describes a model's continuous-time derivatives and outputs.

* `rates`: A named tuple corresponding with the continuous variables, where each field
  contains the rate of change of that continuous variable.
* `outputs`: A named tuple of continuous-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple contains the `RatesOutput` for each submodel.
* `stop`: Set to true to request that the simulation stop after this sample completes.
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

* `updates`: A named tuple corresponding with the discrete variables, where each field
  contains the update of that variable.
* `outputs`: A named tuple of discrete-time outputs (must match the original
  `ModelDescription`).
* `models`: A named tuple contains the `UpdatesOutput` for each submodel.
* `t_next`: The next time at which this model is requesting a stop.
* `stop`: Set to true to request that the simulation stop after this sample completes.
"""
struct UpdatesOutput{UT, OT, MT}
    updates::UT
    outputs::OT
    models::MT
    t_next::Rational{Int64}
    stop::Bool
end
UpdatesOutput(;
    updates = (;),
    outputs = (;),
    models = (;),
    t_next = 0//1,
    stop = false,
) = UpdatesOutput(updates, outputs, models, rationalize(t_next), stop)

"""
    BranchingSeed

This type is useful for making a tree of random seeds that trace back to a single top-level
seed. Here's an example of creating a `BranchingSeed` and creating a random number generator
from it:

```
seed = BranchingSeed(0, "")
rng = Xoshiro(seed)
```

Here is an example of a function that takes in a seed and creates multiple RNGs from it:

```
function foo(seed)

    # Create a top-level branching seed.
    branching_seed = BranchingSeed(seed, "")

    # Model Process A.
    branching_seed_a = branch(branching_seed, "a")
    rng_a = Xoshiro(branching_seed_a)
    x = randn(rng_a, 100)

    # Model Process B.
    branching_seed_b = branch(branching_seed, "b")
    rng_b = Xoshiro(branching_seed_b)
    y = randn(rng_b, 200)

    ...

end
```

In this example, the draws from `rng_a` and `rng_b` are independent of each other, but they
both still change when the top-level seed changes. This allows a user to model separate
random processes, where changing how many random draws are used as part of "process a"
doesn't change the draws of "process b". It's a very useful pattern for making models with
submodels; each submodel can `branch` from its parent's seed according to that model's name.
Then, even if models are swapped for different models, the remaining models will still
generate the same random draws over time.
"""
struct BranchingSeed
    salt::Int64
    breadcrumbs::String
end

"""
    branch(seed::BranchingSeed, name::AbstractString)

Creates a new `BranchingSeed` from the given `seed` by appending the given `name`.
"""
function branch(seed::BranchingSeed, name::AbstractString)
    return BranchingSeed(seed.salt, seed.breadcrumbs * "/" * name)
end

"""
    seed::BranchingSeed / name::AbstractString

This is syntactic sugar for `branch(seed, name)`.
"""
function Base.:/(seed::BranchingSeed, name::AbstractString)
    return branch(seed, name)
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
function VariableDescription{T}(value; title, dimensions, groups = missing, interpolator = missing) where {T}
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

"Creates a Xoshiro RNG from the given BranchingSeed."
Random.Xoshiro(seed::BranchingSeed) = Xoshiro(seed.salt + hash(seed.breadcrumbs))

##################
# User Utilities #
##################

# We don't use these internally; they're helpful modeling tools for users.

"""
    is_regular_step_triggering(t, step, offset = 0//1)

Returns true if `t == n * step + offset`, where `n` is an integer. This is useful for
modeling regularly sampled systems (systems with a constant sample rate). If step == 0, that
means "always triggering".

```
is_regular_step_triggering(10.1, 0.05) # true
is_regular_step_triggering(10.1, 0.20) # false
is_regular_step_triggering(10.1, 0.0) # true
is_regular_step_triggering(10.1, 0.20, 0.1) # true
```
"""
function is_regular_step_triggering(t, step, offset = 0//1)
    return iszero(step) || (mod(rationalize(t - offset), rationalize(step)) == 0//1)
end

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
export ContinuousWhiteNoise
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
export DiscreteWhiteNoise
function (nu::DiscreteWhiteNoise{T})(rng, t) where {T}
    return nu.sigma .* randn(rng, T)
end

#########################
# TypedModelDescription #
#########################

"""
This is the same as ModelDescription, except that any VariableDescription stuff has been
pulled out and all types are fixed as type parameters. This is what's used by the sim loop.
"""
@kwdef struct TypedModelDescription{T, CT, XCT, XDT, YCT, YDT, WCT, WDT, MT}
    type::Type{T} # This could actually be any function that takes kwargs.
    constants::CT
    continuous_states::XCT
    discrete_states::XDT
    continuous_outputs::YCT
    discrete_outputs::YDT
    continuous_random_variables::WCT
    discrete_random_variables::WDT
    models::MT
    t_next::Rational{Int64}
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

function create_typed_model_description(desc::ModelDescription, seed::BranchingSeed)
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
        models = NamedTuple(
            field => create_typed_model_description(
                desc.models[field], seed / string(field),
            )
            for field in fieldnames(typeof(desc.models))
        ),
        t_next = desc.t_next,
    )
end

#########################
# ModelStateDescription #
#########################

# This is our internal representation of the stuff necessary to construct the model form.

@kwdef struct ModelStateDescription{T, CT, XCT, XDT, WCT, WDT, MT}
    constants::CT
    continuous_states::XCT
    discrete_states::XDT
    continuous_random_variables::WCT
    discrete_random_variables::WDT
    models::MT
    t_next::Rational{Int64}
end
function ModelStateDescription{T}(;
    constants = (;),
    continuous_states = (;),
    discrete_states = (;),
    continuous_random_variables = (;),
    discrete_random_variables = (;),
    models = (;),
    t_next = 0//1,
) where {T}
    return ModelStateDescription{
        T, typeof(constants), typeof(continuous_states), typeof(discrete_states),
        typeof(continuous_random_variables), typeof(discrete_random_variables),
        typeof(models)
    }(
        constants, continuous_states, discrete_states,
        continuous_random_variables, discrete_random_variables, models,
        rationalize(t_next),
    )
end

# This has no allocations for bits types.
function model(desc::ModelStateDescription{Nothing})
    return (;
        desc.constants...,
        desc.continuous_states...,
        desc.continuous_random_variables...,
        desc.discrete_states...,
        desc.discrete_random_variables...,
        map(model, desc.models)...,
    )
end

# This has no allocations for bits types.
function model(desc::ModelStateDescription{T}) where {T}
    return T(;
        desc.constants...,
        desc.continuous_states...,
        desc.continuous_random_variables...,
        desc.discrete_states...,
        desc.discrete_random_variables...,
        map(model, desc.models)...,
    )
end

# This has no allocations for bits types.
function copy_model_state_description_except(md::T; kwargs...) where {T <: ModelStateDescription}
    return T(;
        md.constants,
        md.continuous_states,
        md.discrete_states,
        md.continuous_random_variables,
        md.discrete_random_variables,
        md.models,
        md.t_next,
        kwargs...
    )
end

################
# Stop Reasons #
################

abstract type AbstractStopReason end
struct UnknownStopReason <: AbstractStopReason end
struct ReachedEndTime <: AbstractStopReason
    t_end::Rational{Int64}
end
struct ModelRequestedStop <: AbstractStopReason
    model_path::String # What ultimately populates this?
    reason::String
end
struct EncounteredError <: AbstractStopReason
    time::Float64
    exception::Exception
    trace::Any
end

describe(stop::AbstractStopReason) = string(typeof(stop))
describe(stop::UnknownStopReason) = "The sim stopped for an unknown reason."
describe(stop::ReachedEndTime) = "The sim reached the specified end time of $(float(stop.t_end))."
describe(stop::ModelRequestedStop) = "A model ($(stop.model_path)) requested a stop: $(stop.reason)."
describe(stop::EncounteredError) = "The sim experienced an error."

##############
# SimOptions #
##############

include("Logs.jl")
using .Logs

# We define this here so Solvers can import the symbol.
function draw_wc end

include("Solvers.jl")
using .Solvers

include("Hooks.jl")

"""
A set of options for the `simulate` function, with keyword arguments for:

* `log`: Log options to use (e.g., `Logs.BasicLogOptions()`)
* `solver`: Solver to use (e.g., `Solvers.DormandPrince54Options()`)
* `hooks`: A vector of hooks (e.g., `[ProgressBarOptions(),]`)
* `time_dimension`: A `Dimension` for the time unit (e.g., `["time" => "s"]`).
"""
@kwdef struct SimOptions
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
* `step`: The reason the sim stopped

This type acts like a log itself, so for instance these do the same thing:

```
history["/models/plant"]["position"]
history.log["/models/plant"]["position"]
```

The `keys`, `values`, and `pairs` functions also pass through to the underlying log.
"""
struct SimHistory
    model::ModelDescription
    log::AbstractLog
    stop::AbstractStopReason
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

############
# The Loop #
############

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

function draw_wc(t_last, t_next, ommd::TypedModelDescription, msd::ModelStateDescription)
    return copy_model_state_description_except(msd;
        continuous_random_variables = draw_crvs(
            ommd.continuous_random_variables, t_last, t_next,
        ),
        models = NamedTuple{keys(msd.models)}(
            map(ommd.models, msd.models) do ommd_submodel, msd_submodel
                draw_wc(t_last, t_next, ommd_submodel, msd_submodel)
            end
        ),
    )
end

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
        models = NamedTuple(
            mn => create_model_state(t, ommd.models[mn])
            for mn in keys(ommd.models)
        ),
        ommd.t_next,
    )
end

function draw_wd(t, ommd::TypedModelDescription, msd::ModelStateDescription)
    return copy_model_state_description_except(msd;
        discrete_random_variables = draw_drvs(ommd.discrete_random_variables, t),
        models = NamedTuple{keys(msd.models)}(
            map(ommd.models, msd.models) do ommd_submodel, msd_submodel
                draw_wd(t, ommd_submodel, msd_submodel)
            end
        ),
    )
end

function log_continuous_stuff!(t, mh::Nothing, msd::ModelStateDescription, ro::RatesOutput)
end

function log_continuous_stuff!(t, mh, msd::ModelStateDescription, ro::RatesOutput)
    for fn in keys(msd.continuous_states)
        push!(mh.continuous_states[fn], float(t), msd.continuous_states[fn])
    end
    for fn in keys(ro.outputs)
        push!(mh.continuous_outputs[fn], float(t), ro.outputs[fn])
    end
    # TODO: Log the derivatives too.
    for fn in keys(msd.models)
        if haskey(ro.models, fn)
            log_continuous_stuff!(t, mh.models[fn], msd.models[fn], ro.models[fn])
        end
    end
end

function log_discrete_stuff!(t, mh::Nothing, md::TypedModelDescription)
end

function log_discrete_stuff!(t, mh::Nothing, md::UpdatesOutput)
end

# This one is only called during initialization.
function log_discrete_stuff!(t, mh, md::TypedModelDescription)
    for fn in keys(md.discrete_states)
        push!(mh.discrete_states[fn], float(t), md.discrete_states[fn])
    end
    for fn in keys(md.discrete_outputs)
        push!(mh.discrete_outputs[fn], float(t), md.discrete_outputs[fn])
    end
    for fn in keys(md.models)
        log_discrete_stuff!(t, mh.models[fn], md.models[fn])
    end
end

# This is called right after updating.
function log_discrete_stuff!(t, mh, uo::UpdatesOutput)
    # TODO: Log the continuous states too, if those are allowed to change.
    for fn in keys(uo.updates)
        push!(mh.discrete_states[fn], float(t), uo.updates[fn])
    end
    for fn in keys(uo.outputs)
        push!(mh.discrete_outputs[fn], float(t), uo.outputs[fn])
    end
    for fn in keys(uo.models)
        log_discrete_stuff!(t, mh.models[fn], uo.models[fn])
    end
end

function update_discrete_states(discrete_states::T1, updated_discrete_states::T2) where {T1, T2}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(T2, f)
                updated_discrete_states[f]
            else
                discrete_states[f]
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

# If there's no t_next, keep the last one.
function update_model_t_next(last_t_next, updated_t_next)
    iszero(updated_t_next) ? last_t_next : updated_t_next # TODO: How do we want to indicate that there is no new t_next?
end

function update(msd::ModelStateDescription, updates_output::UpdatesOutput)
    return copy_model_state_description_except(
        msd;
        # TODO: Are continuous-time states allowed to change here? Seems like we should allow that.
        discrete_states = update_discrete_states(msd.discrete_states, updates_output.updates),
        models = update_submodels(msd.models, updates_output.models),
        t_next = update_model_t_next(msd.t_next, updates_output.t_next),
    )
end

function find_soonest_t_next_from_models(t_last, msd::ModelStateDescription{T}) where {T}
    t_next_from_this_model = if msd.t_next > t_last
        msd.t_next
    else
        1//0 # If t_next is in the past, it no longer limits us.
    end
    return minimum(
        find_soonest_t_next_from_models(t_last, el) for el in msd.models;
        init = t_next_from_this_model,
    )
end

function step!(mh, t, ommd, rates_fcn, updates_fcn, t_last, msd, solver, hooks, t_end, t_next_suggested)

    # Figure out how big this step can be.

    # Assume the next stop is the next time a user asked for a stop (which might be the end
    # time).
    k_last_requested_stop = searchsortedlast(t, t_last) # Returns that last index of t that is <= t_last
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

    # Get the soonest from what the user asked for, what the integrator suggested, and what
    # the models requested.
    t_next = min(t_next_from_user, t_next_suggested, t_next_from_models)

    # Perform the continuous-time update from t_last to t_next.
    # println("Stepping from $(float(t_last)) to $(float(t_next)).")

    # Potentially, we should draw_wc and keep the end step time no matter what.

    # Step the continuous system. Note that this might not step all the way to the preferred
    # t_next.
    solver_outputs   = solve(ommd, solver, t_last, t_next, msd, rates_fcn, t_end)
    t_next           = solver_outputs.t_completed
    msd              = solver_outputs.msd_k
    stop             = solver_outputs.stop
    t_next_suggested = solver_outputs.t_next_suggested

    # Log the beginning of that sample now that we have its draws and derivatives.
    log_continuous_stuff!(t_last, mh, solver_outputs.msd_km1, solver_outputs.rates)

    # If it's time to stop and nothing else has a reason to stop yet, set the stop reason.
    if isa(stop, UnknownStopReason) && t_last == t_end
        stop = ReachedEndTime(t_end)
    end

    # If there's a reason to stop, bail on the rest of this step.
    if !isa(stop, UnknownStopReason)
        return (t_last, msd, stop, t_next_suggested)
    end

    # Make the discrete draws.
    msd = draw_wd(t_next, ommd, msd)

    # Perform the discrete update from t_next^- to t_next^+.
    updates = updates_fcn(t_next, model(msd))
    msd = update(msd, updates)

    # Log the updated values.
    log_discrete_stuff!(t_next, mh, updates)

    # Update the hooks.
    if !isempty(hooks)
        m = model(msd)
        for hook in hooks
            Hooks.update_hook!(hook, t_next, m) # TODO: Let these stop/pause the loop.
        end
    end

    return (t_next, msd, UnknownStopReason(), t_next_suggested)

end

function loop!(mh, t, ommd, rates_fcn, updates_fcn, msd, solver, hooks)
    t_completed = first(t)
    t_end = last(t)
    t_next_suggested = t_completed + get_initial_time_step(solver)
    stop = UnknownStopReason()
    try
        while isa(stop, UnknownStopReason)
            t_completed, msd, stop, t_next_suggested = step!(
                mh, t, ommd, rates_fcn, updates_fcn, t_completed, msd,
                solver, hooks, t_end, t_next_suggested,
            )
        end
    catch err
        trace = catch_backtrace()
        @error "The simulation encounted an error." exception = (err, trace)
        stop = EncounteredError(float(t_completed), err, stacktrace(trace))
    end
    return (t_completed, msd, stop)
end

############
# simulate #
############

function _initialize(model_description::ModelDescription, seed = 0, t_start = 0//1)

    # Create the top-level branching seed used for default random-variable streams.
    branching_seed = BranchingSeed(seed, "")

    # Now that the time histories are started, we have no further use of the
    # VariableDescriptions. Strip those out for the "original minimal model description".
    # We'll always keep this original description around for its random-variable functions.
    #
    # This is what creates the TypedModelDescription for us.
    ommd = create_typed_model_description(model_description, branching_seed)

    # We can now fill in the draws to have a "model state description".
    msd = create_model_state(t_start, ommd)

    return model(msd)

end

function _initialize(model_prototype; init_fcn, seed = 0, t_start = 0//1)

    # Create the top-level branching seed that will be passed to the user's init function.
    branching_seed = BranchingSeed(seed, "")

    # Run the initialization to get the description of the models given the prototype.
    model_description = init_fcn(t_start, model_prototype, branching_seed)

    # Now that the time histories are started, we have no further use of the
    # VariableDescriptions. Strip those out for the "original minimal model description".
    # We'll always keep this original description around for its random-variable functions.
    #
    # This is what creates the TypedModelDescription for us.
    ommd = create_typed_model_description(model_description, branching_seed)

    # We can now fill in the draws to have a "model state description".
    msd = create_model_state(t_start, ommd)

    # From the model state description, we can build the model itself (the single structure
    # that has fields for all of the variables that were described in the original model
    # description).

    return (; model_description, ommd, msd)

end

"""
    initialize(user_data; init_fcn, seed = 0, t_start = 0)

This is useful for debugging model initialization. Provided the `user_data`, `init_fcn`, and
optionally `seed` and `t_start`, it will run the `init_fcn`, construct the model, and return
it. The `init_fcn` receives a `BranchingSeed` derived from the integer `seed`.
"""
function initialize(model_prototype; kwargs...)
    return model(_initialize(model_prototype; kwargs...).msd)
end

"""
    initialize(model_description; seed = BranchingSeed(0, ""), t_start = 0)

This is useful for debugging model initialization. Given a `ModelDescription` (such as would
be provided by the `init_fcn` input to `simulate`, this will construct and return the model.
The `seed` is used for random variables that do not provide their own
`RandomVariableDescription` seed.
"""
function initialize(
    model_description::ModelDescription;
    seed::BranchingSeed = BranchingSeed(0, ""),
    t_start = 0//1,
)
    ommd = create_typed_model_description(model_description, seed)
    msd = create_model_state(t_start, ommd)
    return model(msd)
end

"""
    simulate(user_data; t, init_fcn, rates_fcn, updates_fcn, close_fcn, seed, options)

Runs a simulation, returning the time history, end time, and final model.

* `user_data`: Can be anything used by the `init_fcn`
* `t`: A collection of monotonic times; the sim will step to exactly each given time, plus
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
    model_prototype;
    t, # Any collection; sim starts at first(t) and goes to last(t) and breaks at everything in between.
    init_fcn, # Turns the prototype into a model description, which can be turned into a model
    rates_fcn = (args...) -> RatesOutput(),
    updates_fcn = (args...) -> UpdatesOutput(),
    close_fcn = (t, model) -> nothing,
    seed = 0,
    options::SimOptions = SimOptions(),
)
    # This might be a tuple with (t_start, t_end), but it can also be any collection of
    # monotonic times.
    t = [rationalize(el) for el in t]
    t_start = first(t)
    t_end = last(t)

    # Pull out the full model description from the initialization function, as well as the
    # typed model description, and finally the model state description.
    model_description, ommd, msd = _initialize(model_prototype; init_fcn, t_start, seed)
    initial_model = model(msd)

    # Use those descriptions to set up the time histories.
    log, mh = create_log(options.log, model_description, options.time_dimension)

    # Log the initial stuff.
    log_discrete_stuff!(t_start, mh, ommd)

    # Create the solver.
    solver = create_solver(options.solver, msd)

    # Create the hooks.
    hooks = map(options.hooks) do hook_options
        return Hooks.create_hook(hook_options, t, initial_model)
    end

    # Propagate until we're done.
    t_end, msd, stop = loop!(mh, t, ommd, rates_fcn, updates_fcn, msd, solver, hooks)

    # Create the final model.
    final_model = model(msd)

    # Close out the models.
    close_fcn(t_end, final_model)

    # Wrap up all of the history into a single object.
    history = SimHistory(model_description, log, stop)

    # Close the hooks.
    for m in hooks
        Hooks.close_hook!(m, t_end, final_model)
    end

    return (history, t_end, final_model)

end

end # module SystemsOfSystems
