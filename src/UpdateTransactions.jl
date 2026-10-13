"""Private runtime collection with complete-tree consumer phase barriers."""
module UpdateTransactions

using ..SystemsOfSystems: SystemsOfSystems, ModelStateDescription, TypedModelDescription,
    UpdatesOutput, KEEP_T_NEXT, exact_time, model, validate_updates_output,
    validate_output_names,
    validate_model_outputs, update, update_states, update_submodels, update_model_t_next,
    copy_model_state_description_except, find_model_requested_stop,
    find_model_requested_stop_in_models, prepend_model_stop_path, first_stop,
    SimulationLogging

abstract type AbstractUpdatePlan end

mutable struct OpaqueUpdateSlot <: AbstractUpdatePlan
    value::Any
end
OpaqueUpdateSlot() = OpaqueUpdateSlot(nothing)

mutable struct CompositionUpdatePlan{K,C <: NamedTuple} <: AbstractUpdatePlan
    kind::K
    children::C
    opaque::OpaqueUpdateSlot
    composed::Bool
    method_world::UInt
    signature::Any
    eligible::Bool
end

function CompositionUpdatePlan(kind, children::NamedTuple)
    return CompositionUpdatePlan(
        kind, children, OpaqueUpdateSlot(), false, UInt(0), nothing, false,
    )
end

# Packages opt in only known composition methods. The fallback neither constructs a user
# model nor invokes a callback to discover its output schema.
build_plan(callback, description, state) = nothing

mutable struct UpdateTransaction{P <: AbstractUpdatePlan}
    plan::P
    leased::Bool
    method_world::UInt
    eligible::Bool
    collections::Int
    compositions::Int
    fallbacks::Int
    description_type::Type
    state_type::Type
end

const EMPTY_PARENT = UpdatesOutput()
const CONSUMERS = (
    exact_time, validate_updates_output, validate_output_names, validate_model_outputs,
    update, update_states, update_submodels, update_model_t_next,
    copy_model_state_description_except,
    find_model_requested_stop, find_model_requested_stop_in_models, prepend_model_stop_path,
    SimulationLogging.log_discrete_stuff!,
    SimulationLogging.log_discrete_event_model!,
    SimulationLogging.log_discrete_event_models!,
    SimulationLogging.log_discrete_state_changes!,
    SimulationLogging.log_continuous_state_updates!,
    SimulationLogging.log_discrete_outputs!,
)
const CONSUMER_METHODS = map(function_value -> Tuple(methods(function_value)), CONSUMERS)
const CONSTRUCTOR_METHODS = Tuple(methods(UpdatesOutput))
const DESCRIPTOR_PROPERTY_METHOD = which(getproperty, Tuple{ModelStateDescription,Symbol})
const TYPED_DESCRIPTION_PROPERTY_METHOD =
    which(getproperty, Tuple{TypedModelDescription,Symbol})

function ordinary_descriptor_properties(description, state)

    which(getproperty, Tuple{typeof(description),Symbol}) ===
        TYPED_DESCRIPTION_PROPERTY_METHOD || return false
    which(getproperty, Tuple{typeof(state),Symbol}) ===
        DESCRIPTOR_PROPERTY_METHOD || return false
    descriptions = getfield(description, :models)
    states = getfield(state, :models)
    keys(descriptions) == keys(states) || return false
    return all(keys(states)) do name
        ordinary_descriptor_properties(getfield(descriptions, name), getfield(states, name))
    end

end

function constructor_keyword_methods()

    # Core.kwcall is shared by every keyword function. Restrict the check to signatures
    # intersecting this particular zero-positional-argument constructor invocation.
    signature = Tuple{typeof(Core.kwcall), NamedTuple, Type{UpdatesOutput}}
    return Tuple(method for method in methods(Core.kwcall)
        if typeintersect(method.sig, signature) !== Union{})

end
const CONSTRUCTOR_KEYWORD_METHODS = constructor_keyword_methods()

function output_property_methods()

    signature = Tuple{typeof(getproperty), UpdatesOutput, Symbol}
    return Tuple(method for method in methods(getproperty)
        if typeintersect(method.sig, signature) !== Union{})

end
const OUTPUT_PROPERTY_METHODS = output_property_methods()

function same_methods(actual, expected)
    return length(actual) == length(expected) && all(actual) do method
        any(original -> original === method, expected)
    end
end

function ordinary_consumers()

    try
        all(zip(CONSUMERS, CONSUMER_METHODS)) do (callback, expected)
            same_methods(Tuple(methods(callback)), expected)
        end || return false
        same_methods(Tuple(methods(UpdatesOutput)), CONSTRUCTOR_METHODS) || return false
        same_methods(constructor_keyword_methods(), CONSTRUCTOR_KEYWORD_METHODS) ||
            return false
        return same_methods(output_property_methods(), OUTPUT_PROPERTY_METHODS)
    catch error
        error isa InterruptException && rethrow()
        return false
    end

end

function make_transaction(callback, description, state)

    plan = try
        build_plan(callback, description, state)
    catch error
        error isa InterruptException && rethrow()
        return nothing
    end
    plan === nothing && return nothing
    return UpdateTransaction(
        plan, false, UInt(0), false, 0, 0, 0, typeof(description), typeof(state),
    )

end

function eligible!(transaction::UpdateTransaction, description, state)

    typeof(description) === transaction.description_type || return false
    typeof(state) === transaction.state_type || return false
    world = Base.get_world_counter()
    if transaction.method_world != world
        transaction.eligible = ordinary_consumers() &&
            ordinary_descriptor_properties(description, state)
        transaction.method_world = world
    end
    return transaction.eligible

end

function selected_composition!(test, plan::CompositionUpdatePlan, signature)

    world = Base.get_world_counter()
    if plan.method_world != world || plan.signature !== signature
        plan.eligible = try
            test()
        catch error
            error isa InterruptException && rethrow()
            false
        end
        plan.method_world = world
        plan.signature = signature
    end
    return plan.eligible

end

function store!(slot::OpaqueUpdateSlot, value)
    slot.value = value
    return nothing
end

function store!(plan::CompositionUpdatePlan, value)
    plan.composed = false
    store!(plan.opaque, value)
    return nothing
end

function clear!(slot::OpaqueUpdateSlot)
    slot.value = nothing
    return nothing
end

@generated function clear_children!(children::C) where {C <: NamedTuple}

    # Emit visits in declaration order so each child's concrete type remains visible to
    # the compiler. Keep clear! dispatch visible for every child.
    calls = map(fieldnames(C)) do name
        :(clear!(getfield(children, $(QuoteNode(name)))))
    end
    return quote
        $(calls...)
        nothing
    end

end

function clear!(plan::CompositionUpdatePlan)
    clear!(plan.opaque)
    clear_children!(plan.children)
    plan.composed = false
    return nothing
end

function cleared(slot::OpaqueUpdateSlot)
    return slot.value === nothing
end

function cleared(plan::CompositionUpdatePlan)
    return !plan.composed && cleared(plan.opaque) && all(cleared, plan.children)
end

function collect!(slot::OpaqueUpdateSlot, callback, args...)
    store!(slot, callback(args...))
    return nothing
end

function collect!(plan::CompositionUpdatePlan, callback, args...)
    store!(plan, callback(args...))
    return nothing
end

validate!(slot::OpaqueUpdateSlot, description) =
    validate_updates_output(description, slot.value)

@generated function validate_children!(children::C, descriptions::D) where {
    C <: NamedTuple, D <: NamedTuple,
}

    calls = map(intersect(fieldnames(D), fieldnames(C))) do name
        field = QuoteNode(name)
        :(validate!(getfield(children, $field), getfield(descriptions, $field)))
    end
    return quote
        $(calls...)
        nothing
    end

end

function validate!(plan::CompositionUpdatePlan, description)

    plan.composed || return validate!(plan.opaque, description)
    validate_output_names(
        EMPTY_PARENT.updates, description.continuous_states,
        description.discrete_states, description.model_path, "UpdatesOutput.updates",
    )
    validate_output_names(
        EMPTY_PARENT.outputs, description.discrete_outputs,
        description.model_path, "UpdatesOutput.outputs",
    )
    validate_output_names(
        plan.children, description.models, description.model_path, "UpdatesOutput.models",
    )
    validate_children!(plan.children, description.models)
    return nothing

end

find_stop(slot::OpaqueUpdateSlot) = find_model_requested_stop(slot.value)

@generated function find_child_stop(children::C) where {C <: NamedTuple}

    calls = map(fieldnames(C)) do name
        field = QuoteNode(name)
        quote
            stop = find_stop(getfield(children, $field))
            isnothing(stop) || return prepend_model_stop_path(stop, $field)
        end
    end
    return quote
        $(calls...)
        nothing
    end

end

function find_stop(plan::CompositionUpdatePlan)
    plan.composed || return find_stop(plan.opaque)
    return find_child_stop(plan.children)
end

apply_state(slot::OpaqueUpdateSlot, state) = update(state, slot.value)

@generated function apply_children(children::C, states::S)::S where {
    C <: NamedTuple, S <: NamedTuple,
}

    values = map(fieldnames(S)) do name
        field = QuoteNode(name)
        hasfield(C, name) ?
            :(apply_state(getfield(children, $field), getfield(states, $field))) :
            :(update(getfield(states, $field), nothing))
    end
    return :(NamedTuple{$(QuoteNode(fieldnames(S)))}(tuple($(values...))))

end

function apply_state(plan::CompositionUpdatePlan, state)

    plan.composed || return apply_state(plan.opaque, state)
    return copy_model_state_description_except(
        state;
        continuous_states = update_states(state.continuous_states, EMPTY_PARENT.updates),
        discrete_states = update_states(state.discrete_states, EMPTY_PARENT.updates),
        models = apply_children(plan.children, state.models),
        t_next = update_model_t_next(state.t_next, KEEP_T_NEXT),
    )

end

function log_events!(slot::OpaqueUpdateSlot, t_f, runtime, prior)
    SimulationLogging.log_discrete_event_model!(t_f, runtime, slot.value, prior)
    return nothing
end

@generated function log_child_events!(children::C, t_f, runtimes::R, priors::P) where {
    C <: NamedTuple, R <: NamedTuple, P <: NamedTuple,
}

    calls = map(intersect(fieldnames(R), fieldnames(C))) do name
        field = QuoteNode(name)
        quote
            runtime = getfield(runtimes, $field)
            if SimulationLogging.sampling_groups_log_sample(
                runtime.sampling_groups_in_subtree,
            )
                log_events!(
                    getfield(children, $field), t_f, runtime, getfield(priors, $field),
                )
            end
        end
    end
    return quote
        $(calls...)
        nothing
    end

end

function log_events!(plan::CompositionUpdatePlan, t_f, runtime, prior)

    plan.composed || return log_events!(plan.opaque, t_f, runtime, prior)
    # This keeps the original parent field order while its empty child set stops the
    # public helper from recursing. Captured children are visited in runtime order below.
    SimulationLogging.log_discrete_event_model!(t_f, runtime, EMPTY_PARENT, prior)
    log_child_events!(plan.children, t_f, runtime.models, prior.models)
    return nothing

end

log_discrete!(plan, t, ::Nothing, prior, updated) = nothing

function log_discrete!(plan, t, runtime, prior, updated)

    groups = runtime.sampling_groups_in_subtree
    SimulationLogging.update_sampling_groups!(t, groups)
    if SimulationLogging.sampling_groups_log_sample(groups)
        log_events!(plan, float(t), runtime, prior)
    end
    if SimulationLogging.sampling_groups_snapshot_states(groups)
        SimulationLogging.log_discrete_snapshot_model!(float(t), runtime, updated)
    end
    return nothing

end

function persistent_update(callback, t, description, state, logging_runtime, stop)

    output = callback(t, model(state))
    validate_updates_output(description, output)
    stop = first_stop(stop, find_model_requested_stop(output))
    updated = update(state, output)
    SimulationLogging.log_discrete_stuff!(
        t, float(t), logging_runtime, output, state, updated,
    )
    return updated, stop

end

consume!(::Nothing, callback, t, description, state, logging_runtime, stop) =
    persistent_update(callback, t, description, state, logging_runtime, stop)

function consume!(
    transaction::UpdateTransaction, callback, t, description, state, logging_runtime, stop,
)

    if transaction.leased
        transaction.fallbacks += 1
        return persistent_update(callback, t, description, state, logging_runtime, stop)
    end
    transaction.leased = true
    try
        if !eligible!(transaction, description, state)
            transaction.fallbacks += 1
            return persistent_update(callback, t, description, state, logging_runtime, stop)
        end
        # Lease starts before construction because user constructors may reenter. All
        # callbacks finish before any validation, stop selection or persistent update.
        value = model(state)
        collect!(transaction.plan, callback, t, value)
        transaction.collections += 1
        if transaction.plan isa CompositionUpdatePlan && transaction.plan.composed
            transaction.compositions += 1
        end
        validate!(transaction.plan, description)
        stop = first_stop(stop, find_stop(transaction.plan))
        updated = apply_state(transaction.plan, state)
        log_discrete!(transaction.plan, t, logging_runtime, state, updated)
        return updated, stop
    finally
        clear!(transaction.plan)
        transaction.leased = false
    end

end

end
