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

    # An omitted derivative hierarchy leaves this entire state subtree unchanged.
    if isempty(rates.rates) && isempty(rates.models)
        return state
    end

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

An empty stage tuple leaves `state` unchanged. Otherwise, all stages must have the same
derivative structure: continuous states and submodels omitted from the first stage must
also be omitted from every later stage.
"""
function propagate(
    state::ModelStateDescription,
    gains::Tuple,
    rates_at_stages::Tuple,
)

    if isempty(rates_at_stages)
        return state
    end

    # Every stage has the same derivative structure, so checking the first is sufficient.
    first_rates = first(rates_at_stages)
    if isempty(first_rates.rates) && isempty(first_rates.models)
        return state
    end

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

#############################
# Provisional State Storage #
#############################

# These methods construct persistent descriptors and user-facing model values. A private
# workspace is safe only while those standard adapters are in use; custom adapters may
# retain a descriptor itself rather than just the model values passed to user callbacks.
const PROVISIONAL_MODEL_METHODS = Tuple(methods(model))
const PROVISIONAL_RATES_METHODS = Tuple(methods(evaluate_rates))
const PROVISIONAL_PROPERTY_METHOD = which(getproperty, Tuple{ModelStateDescription, Symbol})
const PROVISIONAL_RATES_PROPERTY_METHODS = Tuple(
    methods(getproperty, Tuple{RatesOutput, Symbol}),
)
const PROVISIONAL_PROPAGATION_METHODS = (
    Tuple(methods(propagate)),
    Tuple(methods(propagate_set)),
    Tuple(methods(propagate_models)),
    Tuple(methods(propagate_model_values)),
    Tuple(methods(complete_model_rates)),
    Tuple(methods(copy_model_state_description_except)),
    Tuple(methods(weighted_rate)),
    Tuple(methods(propagate_variable)),
)

"""
Private storage for provisional continuous stages belonging to one integrator.

Every descriptor shell and child link is owned by this workspace. Configuration and state
payloads preserve their ordinary shallow aliases. Assigning a field replaces its value;
no user-owned array, returned derivative, or callback model is mutated by this workspace.
The shells must never escape through public propagation or accepted solver results.
"""
mutable struct ProvisionalStateWorkspace{S, P}
    state::S
    active::Bool
    method_world::UInt
    compatible::Bool
    stage_storage::Any
end

function copy_provisional_shells(state::S) where {S <: ModelStateDescription}
    return S(
        state.static,
        state.continuous_states,
        state.discrete_states,
        state.continuous_random_variables,
        state.discrete_random_variables,
        map(copy_provisional_shells, state.models),
        state.t_next,
    )
end

function standard_model_adapter(state::ModelStateDescription)
    selected = which(model, Tuple{typeof(state)})
    property_method = which(getproperty, Tuple{typeof(state), Symbol})
    return property_method === PROVISIONAL_PROPERTY_METHOD &&
        any(method -> method === selected, PROVISIONAL_MODEL_METHODS) &&
        all(standard_model_adapter, state.models)
end

function standard_provisional_adapters(problem::ContinuousProblem, state)

    try
        selected = which(evaluate_rates, Tuple{typeof(problem), Float64, typeof(state)})
        any(method -> method === selected, PROVISIONAL_RATES_METHODS) || return false
        standard_model_adapter(state) || return false

        # Literal stage field reads bypass only the RatesOutput property boundary. Limit
        # inspection to methods intersecting this type, not unrelated Base properties.
        properties = Tuple(methods(getproperty, Tuple{RatesOutput, Symbol}))
        length(properties) == length(PROVISIONAL_RATES_PROPERTY_METHODS) || return false
        properties_unchanged = all(properties) do method
            any(original -> original === method, PROVISIONAL_RATES_PROPERTY_METHODS)
        end
        properties_unchanged || return false
        current = (
            Tuple(methods(propagate)),
            Tuple(methods(propagate_set)),
            Tuple(methods(propagate_models)),
            Tuple(methods(propagate_model_values)),
            Tuple(methods(complete_model_rates)),
            Tuple(methods(copy_model_state_description_except)),
            Tuple(methods(weighted_rate)),
            Tuple(methods(propagate_variable)),
        )

        # Method enumeration order can change when a precompiled module is loaded. Only
        # membership matters: additions or replacements must still reject private reuse.
        return all(zip(current, PROVISIONAL_PROPAGATION_METHODS)) do (methods, standard)
            length(methods) == length(standard) && all(methods) do method
                any(original -> original === method, standard)
            end
        end
    catch error
        # Ambiguous/custom dispatch must fail through its ordinary execution path rather
        # than introducing a new initialization failure from this capability inspection.
        error isa InterruptException && rethrow()
        return false
    end

end

function create_provisional_workspace(problem::ContinuousProblem, state)

    # Reflection occurs during setup, without executing user constructors, rates, or draws.
    # A custom adapter keeps the existing persistent path from its first invocation.
    standard_provisional_adapters(problem, state) || return nothing
    return ProvisionalStateWorkspace{typeof(state), typeof(problem)}(
        copy_provisional_shells(state), false, Base.get_world_counter(), true, nothing,
    )

end

provisional_workspace_available(::Nothing, problem, state) = false

function provisional_workspace_available(
    workspace::ProvisionalStateWorkspace{S, P},
    problem::ContinuousProblem,
    state,
) where {S, P}

    # A nested use of this integrator must not overwrite the outer call's active shells.
    # The persistent path is also the compatibility adapter for a different state shape.
    (workspace.active || typeof(state) !== S || typeof(problem) !== P) && return false

    # Method additions between uses can introduce a retention-capable adapter. Revalidate
    # only when Julia's method world changes; ordinary steps need no descriptor traversal.
    method_world = Base.get_world_counter()
    if method_world != workspace.method_world
        workspace.compatible = standard_provisional_adapters(problem, state)
        workspace.method_world = method_world
    end
    return workspace.compatible

end

# Copy all noncontinuous values from this attempt's beginning state. In particular, prior
# discrete updates may have replaced random values, configuration, or a requested time.
function refresh_provisional_fields!(destination, source, continuous_states)
    setfield!(destination, :static, source.static)
    setfield!(destination, :continuous_states, continuous_states)
    setfield!(destination, :discrete_states, source.discrete_states)
    setfield!(destination, :continuous_random_variables, source.continuous_random_variables)
    setfield!(destination, :discrete_random_variables, source.discrete_random_variables)
    setfield!(destination, :t_next, source.t_next)
    return nothing
end

@generated function refresh_provisional_children!(destination::D, source::S) where {D, S}

    statements = map(fieldnames(S)) do name
        field = QuoteNode(name)
        return :(refresh_provisional_tree!(
            getfield(destination, $field), getfield(source, $field),
        ))
    end
    return Expr(:block, statements..., :(nothing))

end

function refresh_provisional_tree!(destination, source)
    refresh_provisional_fields!(destination, source, source.continuous_states)
    refresh_provisional_children!(destination.models, source.models)
    return destination
end

# Generate only field routing. The same propagate_set/weighted_rate operations as the
# persistent path still define all numerical work, including zero-weight terms and order.
@generated function propagate_provisional_children!(
    destination::D,
    source::S,
    gains::G,
    stages::R,
) where {D <: NamedTuple, S <: NamedTuple, G <: Tuple, R <: Tuple}

    statements = map(fieldnames(S)) do name
        field = QuoteNode(name)
        rates = map(1:fieldcount(R)) do index
            models_type = fieldtype(fieldtype(R, index), :models)
            if hasfield(models_type, name)
                return :(getfield(getfield(getfield(stages, $index), :models), $field))
            end
            return :(RatesOutput())
        end
        return :(propagate_provisional!(
            getfield(destination, $field), getfield(source, $field),
            gains, $(Expr(:tuple, rates...)),
        ))
    end
    return Expr(:block, statements..., :(nothing))

end

function propagate_provisional!(
    destination::ModelStateDescription,
    source::ModelStateDescription,
    gains::Tuple,
    stages::Tuple,
)

    if isempty(stages) || (isempty(first(stages).rates) && isempty(first(stages).models))
        return refresh_provisional_tree!(destination, source)
    end

    # Parent arithmetic precedes child arithmetic, as in persistent propagation. Child
    # links remain owned by the workspace even when a derivative subtree is omitted.
    continuous_states = propagate_set(
        source.continuous_states, gains, map(stage -> stage.rates, stages),
    )
    refresh_provisional_fields!(destination, source, continuous_states)
    propagate_provisional_children!(destination.models, source.models, gains, stages)
    return destination

end

function provisional_state(::Nothing, source, gains, stages)
    return propagate(source, gains, stages)
end

function provisional_state(workspace::ProvisionalStateWorkspace, source, gains, stages)
    return propagate_provisional!(workspace.state, source, gains, stages)
end

#####################
# Typed Stage Access #
#####################

# These generation-time helpers describe field routing only. Every runtime derivative
# remains in its original stage slot, with the publisher's existing payload references.
stage_element_type(::Type{S}, index) where {S <: Tuple} = fieldtype(S, index)
stage_element_type(::Type{Vector{D}}, index) where {D} = D

function stage_node_type(type, path)

    for name in path
        models = fieldtype(type, :models)
        hasfield(models, name) || return nothing
        type = fieldtype(models, name)
    end
    return type

end

function stage_node_expression(type, index, path)

    root = stage_element_type(type, index)
    stage_node_type(root, path) === nothing && return :(RatesOutput())
    value = type <: Tuple ? :(getfield(stages, $index)) : :(getindex(stages, $index))
    for name in path
        value = :(getfield(getfield($value, :models), $(QuoteNode(name))))
    end
    return value

end

function stage_subtree_is_empty(type, path)

    node = stage_node_type(stage_element_type(type, 1), path)
    return node === nothing ||
        (isempty(fieldnames(fieldtype(node, :rates))) &&
            isempty(fieldnames(fieldtype(node, :models))))

end

@generated function accessed_continuous_states(
    states::X,
    gains::G,
    stages::R,
    ::Val{P},
) where {X <: NamedTuple, G <: Tuple, R, P}

    first_node = stage_node_type(stage_element_type(R, 1), P)
    rate_names = first_node === nothing ? () : fieldnames(fieldtype(first_node, :rates))
    values = map(fieldnames(X)) do name
        field = QuoteNode(name)
        name in rate_names || return :(getfield(states, $field))
        terms = map(1:fieldcount(G)) do index
            node = stage_node_expression(R, index, P)
            return :(getfield(gains, $index) * getfield(getfield($node, :rates), $field))
        end
        # Build the original right-recursive weighted sum, including every zero term.
        # Julia evaluates each left multiplication before entering the right expression.
        combination = last(terms)
        for index in (length(terms) - 1):-1:1
            combination = :($(terms[index]) + $combination)
        end
        original = gensym(:state_value)
        return quote
            $original = getfield(states, $field)
            ($original + $combination)::typeof($original)
        end
    end
    return :(NamedTuple{$(QuoteNode(fieldnames(X)))}($(Expr(:tuple, values...))))

end

@generated function accessed_provisional_children!(
    destination::D,
    source::S,
    gains,
    stages,
    ::Val{P},
) where {D <: NamedTuple, S <: NamedTuple, P}

    calls = map(fieldnames(S)) do name
        field = QuoteNode(name)
        path = QuoteNode((P..., name))
        return :(accessed_provisional!(
            getfield(destination, $field), getfield(source, $field),
            gains, stages, Val{$path}(),
        ))
    end
    return Expr(:block, calls..., :(nothing))

end

@generated function accessed_provisional!(
    destination::D,
    source::S,
    gains::G,
    stages::R,
    path::Val{P},
) where {D <: ModelStateDescription, S <: ModelStateDescription, G <: Tuple, R, P}

    if isempty(fieldnames(G)) || stage_subtree_is_empty(R, P)
        return :(refresh_provisional_tree!(destination, source))
    end
    return quote
        continuous_states = accessed_continuous_states(source.continuous_states,
            gains, stages, path)
        refresh_provisional_fields!(destination, source, continuous_states)
        accessed_provisional_children!(
            destination.models, source.models, gains, stages, path,
        )
        destination
    end

end

function accessed_provisional_state(
    workspace::ProvisionalStateWorkspace, source, gains, stages,
)
    return accessed_provisional!(workspace.state, source, gains, stages, Val(()))
end

@generated function accessed_persistent_children(
    source::S,
    gains,
    stages,
    ::Val{P},
) where {S <: NamedTuple, P}

    values = map(fieldnames(S)) do name
        field = QuoteNode(name)
        path = QuoteNode((P..., name))
        return :(accessed_persistent_state(
            getfield(source, $field), gains, stages, Val{$path}(),
        ))
    end
    return :(NamedTuple{$(QuoteNode(fieldnames(S)))}($(Expr(:tuple, values...))))

end

@generated function accessed_persistent_state(
    source::S,
    gains::G,
    stages::R,
    path::Val{P},
) where {S <: ModelStateDescription, G <: Tuple, R, P}

    if isempty(fieldnames(G)) || stage_subtree_is_empty(R, P)
        return :source
    end
    return quote
        continuous_states = accessed_continuous_states(source.continuous_states,
            gains, stages, path)
        models = accessed_persistent_children(source.models, gains, stages, path)
        copy_model_state_description_except(source; continuous_states, models)
    end

end

function accessed_persistent_state(source, gains, stages)
    return accessed_persistent_state(source, gains, stages, Val(()))
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
