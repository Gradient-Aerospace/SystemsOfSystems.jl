"""
The `ContinuousProblems` module adapts the hierarchical state used by SystemsOfSystems to
the small set of mathematical operations required by a continuous-time integrator.

This boundary is intentionally internal. Model authors work with `ModelDescription`,
`RatesOutput`, and the model passed to `rates_fcn`; solver authors work with operations such
as `draw_continuous_random_variables`, `evaluate_rates`, and `propagate`. Neither side needs
to understand the other's representation.

Keeping these operations out of the solver implementations serves two purposes. First, a
new numerical method does not need to know how constants, discrete state, random variables,
resources, or nested models are stored. Second, all solvers share exactly the same state
semantics, so a correction or optimization to hierarchical propagation benefits every
method.
"""
module ContinuousProblems

using ..SystemsOfSystems: ModelStateDescription, RatesOutput,
    copy_model_state_description_except, draw_wc, model, normalized_variable_error,
    validate_rates_output

"""
    ContinuousProblem(description, rates_fcn)

A container adapting the continuous-time portion of a SystemsOfSystems simulation to the
solver interface.

`typed_description` contains the initialized description needed to prepare random variables.
`rates_fcn` is the user's continuous-time model function. `error_policy` is a concrete
policy for traversing and normalizing local error. All three fields are type parameters so
calls through this adapter remain fully specialized in the numerical solver's inner loop.
"""
struct ContinuousProblem{TD, F, EP}
    typed_description::TD
    rates_fcn::F
    error_policy::EP
end

"""
The initial componentwise normalized-error policy.

This is deliberately an isbits marker rather than a dynamically typed `ModelDescription`.
Future variable metadata can be compiled into a concrete hierarchy of normalization
functions during initialization without burdening every numerical stage.
"""
struct DefaultErrorPolicy end

ContinuousProblem(typed_description, rates_fcn) =
    ContinuousProblem(typed_description, rates_fcn, DefaultErrorPolicy())

"""
    draw_continuous_random_variables(problem, t_start, dt_f, state)

Draws every continuous random variable for a newly committed interval beginning at the
official `t_start` with floating-point duration `dt_f`. Rejected numerical attempts and
accepted substeps within that interval reuse the returned state.
"""
function draw_continuous_random_variables(
    problem::ContinuousProblem,
    t_start,
    dt_f::Float64,
    state::ModelStateDescription,
)
    return draw_wc(
        t_start,
        dt_f,
        problem.typed_description,
        state,
    )
end

"""
    evaluate_rates(problem, t, state)

Construct the model visible to user code and evaluate its continuous-time function at the
floating-point stage time `t`.

The returned `RatesOutput` contains both derivatives and continuous outputs. The first
evaluation from an accepted step is retained for logging and stop requests. Evaluations at
intermediate Runge-Kutta stages are provisional numerical values; their outputs and stop
requests are intentionally ignored by the simulation loop.
"""
function evaluate_rates(
    problem::ContinuousProblem,
    t::AbstractFloat,
    state::ModelStateDescription,
)
    output = problem.rates_fcn(t, model(state))
    validate_rates_output(problem.typed_description, output)
    return output
end

# These helpers apply one derivative to one hierarchical state. A RatesOutput is allowed to
# omit continuous states and submodels that have no dynamics. The state structure, rather
# than the RatesOutput structure, therefore drives every traversal.

function propagate_variable(x::T, gain, rate::T) where {T}
    return (x + gain * rate)::T
end

function propagate_set(states::T1, gain, rates::T2) where {T1, T2}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do field
            if hasfield(T2, field)
                propagate_variable(states[field], gain, rates[field])
            else
                states[field]
            end
        end
    )
end

function complete_model_rates(submodels::NamedTuple, model_rates::NamedTuple)
    return NamedTuple{fieldnames(typeof(submodels))}(
        map(fieldnames(typeof(submodels))) do field
            if hasfield(typeof(model_rates), field)
                model_rates[field]
            else
                RatesOutput()
            end
        end
    )
end

function propagate_models(submodels::NamedTuple, gain, model_rates::NamedTuple)
    complete_rates = complete_model_rates(submodels, model_rates)
    return map(
        (submodel, rates) -> propagate(submodel, gain, rates),
        submodels,
        complete_rates,
    )
end

"""
    propagate(state, gain, rates)

Returns a copy of `state` whose continuous states have been advanced by `gain * rates`.
Everything that is not continuous state is preserved exactly.

This operation is useful for algorithms or sparse tableaus that need only one derivative.
The tuple overload performs a complete linear combination in one hierarchical traversal.
"""
function propagate(
    state::ModelStateDescription,
    gain,
    rates::RatesOutput,
)
    return copy_model_state_description_except(
        state;
        continuous_states = propagate_set(
            state.continuous_states,
            gain,
            rates.rates,
        ),
        models = propagate_models(state.models, gain, rates.models),
    )
end

# These helpers apply a statically sized linear combination of derivatives. Tuple recursion
# keeps the number and concrete types of stages visible to the compiler. It also avoids the
# temporary tuples and broadcast machinery used by the original general propagation path.

function weighted_rate(gains::Tuple{G}, rates::Tuple{R}) where {G, R}
    return first(gains) * first(rates)
end

function weighted_rate(gains::Tuple, rates::Tuple)
    return first(gains) * first(rates) + weighted_rate(Base.tail(gains), Base.tail(rates))
end

function propagate_variable(x::T, gains::Tuple, rates::Tuple) where {T}
    return (x + weighted_rate(gains, rates))::T
end

function propagate_set(states::T1, gains::Tuple, rates_at_stages::Tuple) where {T1}
    first_rates = first(rates_at_stages)
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do field
            if hasfield(typeof(first_rates), field)
                rates = map(stage_rates -> getfield(stage_rates, field), rates_at_stages)
                propagate_variable(states[field], gains, rates)
            else
                states[field]
            end
        end
    )
end

function complete_model_rates(
    submodels::NamedTuple,
    model_rates_at_stages::Tuple,
)
    return map(model_rates_at_stages) do model_rates
        complete_model_rates(submodels, model_rates)
    end
end

# Expanding the submodel tuple at generation time avoids both a many-input map and recursive
# Base.tail specialization. The generated body contains one ordinary propagate call per
# submodel; the hierarchical propagation logic remains in `propagate`.
@generated function propagate_model_values(
    submodels::S,
    gains::G,
    model_rates_at_stages::R,
) where {S <: NamedTuple, G <: Tuple, R <: Tuple}

    names = fieldnames(S)
    values = map(enumerate(names)) do (index, name)
        rates = Expr(
            :tuple,
            map(1:fieldcount(R)) do stage_index
                stage_type = fieldtype(R, stage_index)
                if hasfield(stage_type, name)
                    return :(
                        getfield(
                            getfield(model_rates_at_stages, $stage_index),
                            $(QuoteNode(name)),
                        )
                    )
                else
                    return :(RatesOutput())
                end
            end...,
        )
        return :(
            propagate(
                getfield(submodels, $index),
                gains,
                $rates,
            )
        )
    end
    return :(
        NamedTuple{$(QuoteNode(names))}(
            $(Expr(:tuple, values...)),
        )
    )

end

function propagate_models(
    submodels::NamedTuple,
    gains::Tuple,
    model_rates_at_stages::Tuple,
)
    return propagate_model_values(submodels, gains, model_rates_at_stages)
end

"""
    propagate(state, gains, rates_at_stages)

Returns a copy of `state` advanced by the linear combination of `rates_at_stages` described
by `gains`.

The stage tuple is expected to have a stable derivative structure: if the first stage omits
a continuous state, later stages must omit it as well. This is the same modeling contract as
the original solvers, made explicit here so it can eventually be validated during solver
initialization rather than repeatedly inside the numerical loop.
"""
function propagate(
    state::ModelStateDescription,
    gains::Tuple,
    rates_at_stages::Tuple,
)
    return copy_model_state_description_except(
        state;
        continuous_states = propagate_set(
            state.continuous_states,
            gains,
            map(rates -> rates.rates, rates_at_stages),
        ),
        models = propagate_models(
            state.models,
            gains,
            map(rates -> rates.models, rates_at_stages),
        ),
    )
end

# The adaptive controller receives one normalized error per continuous variable through
# the public `normalized_variable_error` interface. This adapter traverses the continuous
# state hierarchy and takes the maximum across variables and submodels.

"""
    normalized_error(problem, state, embedded_state, absolute_tolerance, relative_tolerance)

Returns the largest fraction of its allowable local error used by any continuous-state
component. A value no greater than one is acceptable to the default adaptive controller.

The problem's error policy can eventually contain the concrete hierarchy of custom
normalization functions constructed from user variable metadata. The initial componentwise
policy traverses the typed description so this performance-critical recursion remains fully
specialized.
"""
function normalized_error(
    problem::ContinuousProblem,
    state::ModelStateDescription,
    embedded_state::ModelStateDescription,
    absolute_tolerance,
    relative_tolerance,
)
    return normalized_error(
        problem.error_policy,
        problem.typed_description,
        state,
        embedded_state,
        absolute_tolerance,
        relative_tolerance,
    )
end

# Descend through the typed model description together with the state hierarchy. The initial
# componentwise policy does not inspect `description`, but keeping the matching description
# at every level is what will let a variable select its own normalized-error function later.
function normalized_error(
    policy,
    description,
    state::ModelStateDescription,
    embedded_state::ModelStateDescription,
    absolute_tolerance,
    relative_tolerance,
)
    errors = map(state.continuous_states, embedded_state.continuous_states) do xc1, xc2
        normalized_variable_error(
            xc1, xc2, absolute_tolerance, relative_tolerance,
        )
    end
    max_error = maximum(errors; init = 0.)
    errors = map(
        description.models, state.models, embedded_state.models,
    ) do subdescription, substate, embedded_substate
        normalized_error(
            policy,
            subdescription,
            substate,
            embedded_substate,
            absolute_tolerance,
            relative_tolerance,
        )
    end
    max_error = maximum(errors; init = max_error)
    return max_error
end

end # ContinuousProblems
