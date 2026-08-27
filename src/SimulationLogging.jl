"""
The `SimulationLogging` module records model states and outputs during the simulation loop.

`Logs` owns storage and compiles model samplers into shared sampling groups.
`SimulationLogging` evaluates those groups and routes each accepted simulation sample
through four distinct paths:

* Continuous state logging reads states from `ModelStateDescription`.
* Continuous output logging reads outputs from `RatesOutput`.
* Discrete event logging reads sparse state changes and outputs from `UpdatesOutput`.
* Discrete snapshot logging reads complete post-update states from
  `ModelStateDescription`.

This module is internal. The simulation setup and loop call its four top-level
`log_..._stuff!` functions; the remaining helpers implement type-stable tree traversal.
"""
module SimulationLogging

using ..SimulationTimes: ExactTime
using ..SystemsOfSystems:
    ModelStateDescription, RatesOutput, TypedModelDescription, UpdatesOutput
import ..Logs

###################
# Sampling Groups #
###################

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

######################
# Continuous Logging #
######################

log_continuous_state_stuff!(::ExactTime, ::Float64, ::Nothing, ::ModelStateDescription) =
    nothing

log_continuous_output_stuff!(::Float64, ::Nothing, ::RatesOutput) = nothing

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

function log_continuous_state_model! end
function log_continuous_output_model! end

# Generate direct field accesses for the heterogeneous child histories. A runtime Symbol
# index makes Julia box those values before the recursive call. The generated code only
# handles this static routing; the logging behavior remains in the model helpers.
@generated function log_continuous_state_models!(
    t_f,
    mh_models::MHT,
    msd_models::MSDT,
) where {
    MHT <: NamedTuple,
    MSDT <: NamedTuple,
}

    statements = map(fieldnames(MHT)) do fn
        field = QuoteNode(fn)
        return quote
            model_history = getfield(mh_models, $field)
            if sampling_groups_log_sample(model_history.sampling_groups_in_subtree)
                log_continuous_state_model!(
                    t_f, model_history, getfield(msd_models, $field),
                )
            end
        end
    end
    return quote
        $(statements...)
        nothing
    end

end

function log_continuous_state_model!(
    t_f::Float64,
    mh::Logs.ModelHistory,
    msd::ModelStateDescription,
)

    # The compiled group contains this model's independently evaluated sampling decision.
    sampling_group = mh.sampling_group

    if sampling_group.log_states
        log_continuous_states!(t_f, mh.continuous_states, msd.continuous_states)
    end
    log_continuous_state_models!(t_f, mh.models, msd.models)

end

function log_continuous_state_stuff!(
    t::ExactTime, t_f::Float64,
    mh::Logs.ModelHistory,
    msd::ModelStateDescription,
)

    # Evaluate each distinct sampler once, then reject the whole tree when every group is
    # inactive at this time.
    sampling_groups = mh.sampling_groups_in_subtree
    update_sampling_groups!(t, sampling_groups)
    if sampling_groups_log_sample(sampling_groups)
        log_continuous_state_model!(t_f, mh, msd)
    end

end

@generated function log_continuous_output_models!(
    t_f,
    mh_models::MHT,
    ro_models::ROT,
) where {
    MHT <: NamedTuple,
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
                log_continuous_output_model!(t_f, model_history, $rates_output)
            end
        end
    end
    return quote
        $(statements...)
        nothing
    end

end

function log_continuous_output_model!(
    t_f::Float64,
    mh::Logs.ModelHistory,
    ro::RatesOutput,
)

    if mh.sampling_group.log_outputs
        log_continuous_outputs!(t_f, mh.continuous_outputs, ro.outputs)
    end
    # TODO: Log the derivatives too.
    log_continuous_output_models!(t_f, mh.models, ro.models)

end

function log_continuous_output_stuff!(
    t_f::Float64,
    mh::Logs.ModelHistory,
    ro::RatesOutput,
)

    # The state phase evaluated each sampler. Reuse those decisions so one logical sample
    # cannot select a state and output differently.
    if sampling_groups_log_sample(mh.sampling_groups_in_subtree)
        log_continuous_output_model!(t_f, mh, ro)
    end

end

###########################
# Initial Discrete Values #
###########################

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

##########################
# Discrete Event Logging #
##########################

# Discrete logging has two distinct sources:
#
# * Events come from UpdatesOutput. CompleteSampler records sparse state changes and
#   discrete outputs this way.
# * Snapshots come from the post-update ModelStateDescription. RegularSampler records every
#   selected discrete state this way, whether or not it changed in the current update.

function log_discrete_stuff!(
    ::ExactTime, ::Float64,
    ::Nothing,
    ::Union{Nothing, UpdatesOutput},
    ::ModelStateDescription,
    ::ModelStateDescription,
)
end

function log_discrete_state_changes!(t_f, mh_xd, uo_updates)
    for fn in fieldnames(typeof(mh_xd))
        if hasfield(typeof(uo_updates), fn)
            push!(mh_xd[fn], t_f, uo_updates[fn])
        end
    end
end

function log_continuous_state_updates!(t_f, mh_xc, uo_updates, prior_xc)
    for fn in fieldnames(typeof(mh_xc))
        if hasfield(typeof(uo_updates), fn)
            push!(mh_xc[fn], t_f, prior_xc[fn])
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

# A present parent UpdatesOutput may explicitly contain `nothing` for a child that produced
# no updates or outputs. There is no event to record for that child; snapshot traversal is
# handled separately from the complete post-update model state.
function log_discrete_event_model!(
    ::Float64,
    ::Logs.ModelHistory,
    ::Nothing,
    ::ModelStateDescription,
)
end

# As in continuous logging, generate only the direct field routing needed to preserve each
# heterogeneous child type. Event traversal follows only the models present in the current
# UpdatesOutput; missing branches contain no state changes or outputs to record.
@generated function log_discrete_event_models!(
    t_f,
    mh_models::MHT,
    uo_models::UOT,
    prior_models::PT,
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
        # Record the *prior* value at `t`. The continuous logger records the updated value
        # in its next state phase, which also occurs at `t`.
        log_continuous_state_updates!(
            t_f, mh.continuous_states, uo.updates, prior.continuous_states,
        )
    end

    # Log whatever outputs they provided this time.
    if sampling_group.log_outputs
        log_discrete_outputs!(t_f, mh.discrete_outputs, uo.outputs)
    end

    log_discrete_event_models!(t_f, mh.models, uo.models, prior.models)

end

#############################
# Discrete Snapshot Logging #
#############################

function log_discrete_state_snapshot!(t_f, mh_xd, updated_xd)
    for fn in fieldnames(typeof(mh_xd))
        push!(mh_xd[fn], t_f, updated_xd[fn])
    end
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

################################
# Discrete Logging Entry Point #
################################

# This is the top-level entry point called right after updating.
function log_discrete_stuff!(
    t::ExactTime, t_f::Float64,
    mh::Logs.ModelHistory,
    uo::Union{Nothing, UpdatesOutput},
    prior::ModelStateDescription,
    updated::ModelStateDescription,
)

    sampling_groups = mh.sampling_groups_in_subtree
    update_sampling_groups!(t, sampling_groups)

    # Event logging sees the sparse update result and pre-update continuous states. A
    # top-level `nothing` means that no event exists, but it does not suppress the separate
    # state snapshot opportunity below.
    if !isnothing(uo) && sampling_groups_log_sample(sampling_groups)
        log_discrete_event_model!(
            t_f, mh, uo, prior,
        )
    end

    # Snapshot logging sees the authoritative post-update state and does not inspect the
    # sparse UpdatesOutput at all.
    if sampling_groups_snapshot_states(sampling_groups)
        log_discrete_snapshot_model!(t_f, mh, updated)
    end

end

end # SimulationLogging
