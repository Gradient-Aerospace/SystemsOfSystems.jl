# TODO:
#
# - [x] Create the Logger interface.
# - [x] Figure out performance.
# - [x] Use an HDF5Logger.
# - [x] Load an HDF5 log.
# - [x] Save constants too.
# - [x] Figure out the time dimension.
# - [x] Move HDF5Log to an extension.
# - [x] Make functions for plotting time series.
# - [x] Make the loop a function.
# - [x] Add a try-catch.
# - [x] Encapsulate the RK4 solver.
# - [x] Make DP54.
# - [x] Tidy up solver interface.
# - [x] Implement a progress bar.
# - [x] Make a NullLog.
# - [x] Add a close_fcn.
# - [ ] Figure out how to log the RNG state in a way we could load later.
# - [ ] Figure out how to capture console output.
# - [x] Attach a license.

module SystemsOfSystems

export simulate, SimOptions, Solvers, Monitors, Logs
export ModelDescription, VariableDescription, is_regular_step_triggering
export RatesOutput, UpdatesOutput

using Random: Xoshiro

include("TimeSeries.jl")
using .TimeSeriesStuff

#########################
# User Function Outputs #
#########################

# AKA InitOutput
"""
TODO
"""
struct ModelDescription{T, CT, XCT, XDT, YCT, YDT, WCT, WDT, MT}
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
    kwargs...
) = ModelDescription(
    type, constants,
    continuous_states, discrete_states,
    continuous_outputs, discrete_outputs,
    continuous_random_variables, discrete_random_variables,
    models,
    rationalize(t_next),
)

"""
TODO
"""
struct RatesOutput{RT, OT, MT}
    rates::RT
    outputs::OT # Should this be continuous_outputs?
    models::MT
    t_next::Rational{Int64}
    stop::Bool # Or stop reason. TODO: I don't love the allocations here. This rest of this can be a bits type.
end
RatesOutput(;
    rates = (;),
    outputs = (;),
    models = (;),
    t_next = 0//1, # TODO: Why is this here?
    stop = false,
) = RatesOutput(rates, outputs, models, rationalize(t_next), stop)

"""
TODO
"""
struct UpdatesOutput{UT, OT, MT}
    updates::UT
    outputs::OT # Should this be discrete_outputs?
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
TODO
"""
struct VariableDescription{T}
    value::T
    title::String
    dimensions::Vector{Dimension}
    # record::Bool # To let users decide on a per-model basis if they want this model to log anything at all. Or, let this be a set of symbols to record.
    VariableDescription(value, title, dimensions) = new{typeof(value)}(value, title, dimensions)
    VariableDescription(value; title, dimensions) = new{typeof(value)}(value, title, Dimension[dimensions...])
    VariableDescription{T}(value; title, dimensions) where {T} = new{T}(value, title, Dimension[dimensions...])
end

strip_fluff_from_variable(var) = var
strip_fluff_from_variable(var::VariableDescription) = var.value

function strip_fluff_from_model_description(desc::ModelDescription)
    return ModelDescription(;
        type = desc.type,
        constants = map(strip_fluff_from_variable, desc.constants),
        continuous_states = map(strip_fluff_from_variable, desc.continuous_states),
        discrete_states = map(strip_fluff_from_variable, desc.discrete_states),
        continuous_outputs = map(strip_fluff_from_variable, desc.continuous_outputs),
        discrete_outputs = map(strip_fluff_from_variable, desc.discrete_outputs),
        continuous_random_variables = map(strip_fluff_from_variable, desc.continuous_random_variables),
        discrete_random_variables = map(strip_fluff_from_variable, desc.discrete_random_variables),
        models = map(strip_fluff_from_model_description, desc.models),
        t_next = desc.t_next,
    )
end

##################
# User Utilities #
##################

# We don't use these internally; they're helpful modeling tools for users.

"""
TODO
"""
function is_regular_step_triggering(t, step, offset = 0//1)
    return mod(rationalize(t + offset), rationalize(step)) == 0//1
end

#########################
# ModelStateDescription #
#########################

# This is our internal representation of the stuff necessary to construct the model form.

# TODO: Does it ever make sense to construct this without all of these arguments? (Possibly
# not continuous_random_variables.)
#
# TODO: Should all of these named tuples have parameters for types?
@kwdef struct ModelStateDescription{T, CT, XCT, XDT, WCT, WDT, MT}
    constants::CT
    continuous_states::XCT
    discrete_states::XDT
    continuous_random_variables::WCT
    discrete_random_variables::WDT
    models::MT
    t_next::Rational{Int64}
end
ModelStateDescription{T}(;
    constants = (;),
    continuous_states = (;),
    discrete_states = (;),
    continuous_random_variables = (;),
    discrete_random_variables = (;),
    models = (;),
    t_next = 0//1,
) where {T} = ModelStateDescription{T, typeof(constants), typeof(continuous_states), typeof(discrete_states), typeof(continuous_random_variables), typeof(discrete_random_variables), typeof(models)}(
    constants, continuous_states, discrete_states,
    continuous_random_variables, discrete_random_variables, models,
    rationalize(t_next),
)

# This has no allocations for bits types.
function model(desc::ModelStateDescription{Nothing})
    return (;
        desc.constants...,
        desc.continuous_states...,
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
describe(stop::ReachedEndTime) = "The sim reached the specified end time $(float(stop.t_end))."
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

include("Monitors.jl")

"""
TODO
"""
@kwdef struct SimOptions
    log::Union{Nothing, Logs.AbstractLogOptions} = Logs.BasicLogOptions()
    solver::Solvers.AbstractSolverOptions = Solvers.DormandPrince54Options()
    monitors::Vector{Monitors.AbstractMonitorOptions} = []
    time_dimension::Dimension = Dimension("time", "s")
    # catch_error::Bool = true
end

##############
# SimHistory #
##############

"""
TODO
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

######################
# Internal Utilities #
######################

function recursively_reduce(op, desc, value) # desc is anything with .models.
    for m in desc.models # Run on submodels first.
        value = recursively_reduce(op, m, value)
    end
    return op(desc, value) # Now do this model.
end

# function recursive_map(f, desc)
#     return (;
#         f(desc)...,
#         models = map(m -> recursive_map(f, m), desc.models),
#     )
# end

############
# The Loop #
############

function draw_wc(t_last, t_next, ommd::MDT, msd::MSDT) where {MDT <: ModelDescription, MSDT <: ModelStateDescription}
    return copy_model_state_description_except(msd;
        continuous_random_variables = map(drvf -> drvf(t_last, t_next), ommd.continuous_random_variables),
        models = NamedTuple{fieldnames(fieldtype(MSDT, :models))}(
            map(ommd.models, msd.models) do ommd_submodel, msd_submodel
                draw_wc(t_last, t_next, ommd_submodel, msd_submodel)
            end
        ),
    )
end

function draw_wd(t, ommd::ModelDescription{T}, md::ModelDescription) where {T}
    return ModelStateDescription{T}(;
        md.constants,
        md.continuous_states,
        md.discrete_states,
        md.continuous_random_variables,
        discrete_random_variables = map(drvf -> drvf(t), ommd.discrete_random_variables),
        models = NamedTuple(
            mn => draw_wd(t, ommd.models[mn], md.models[mn])
            for mn in keys(ommd.models)
        ),
        md.t_next,
    )
end

function draw_wd(t, ommd::MDT, msd::MSDT) where {MDT <: ModelDescription, MSDT <: ModelStateDescription}
    return copy_model_state_description_except(msd;
        discrete_random_variables = map(drvf -> drvf(t), ommd.discrete_random_variables),
        models = NamedTuple{fieldnames(fieldtype(MSDT, :models))}(
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

function log_discrete_stuff!(t, mh::Nothing, md::ModelDescription)
end
function log_discrete_stuff!(t, mh::Nothing, uo::UpdatesOutput)
end

# This one is only called during initialization.
function log_discrete_stuff!(t, mh, md::ModelDescription)
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

# function update(msd::ModelStateDescription{T}, updates_output) where {T}
#     return copy_model_state_description_except(
#         msd;
#         # TODO: Are continuous-time states allowed to change here? Seems like we should allow that.
#         discrete_states = NamedTuple(
#             f => haskey(updates_output.updates, f) ? updates_output.updates[f] : msd.discrete_states[f]
#             for f in keys(msd.discrete_states)
#         ),
#         models = NamedTuple(
#             f => haskey(updates_output.models, f) ? update(msd.models[f], updates_output.models[f]) : msd.models[f]
#             for f in keys(msd.models)
#         ),
#         t_next = iszero(updates_output.t_next) ? msd.t_next : rationalize(updates_output.t_next), # If there's no t_next, keep the last one.
#     )
# end

function update(msd::MSDT, updates_output) where {MSDT <: ModelStateDescription}
    return copy_model_state_description_except(
        msd;
        # TODO: Are continuous-time states allowed to change here? Seems like we should allow that.
        # This does not seem to allocate for bits types:
        discrete_states = NamedTuple{fieldnames(fieldtype(MSDT, :discrete_states))}(
            map(fieldnames(fieldtype(MSDT, :discrete_states))) do f
                haskey(updates_output.updates, f) ? updates_output.updates[f] : msd.discrete_states[f]
            end
        ),
        models = NamedTuple{fieldnames(fieldtype(MSDT, :models))}(
            map(fieldnames(fieldtype(MSDT, :models))) do f
                haskey(updates_output.models, f) ?
                    update(msd.models[f], updates_output.models[f]) :
                    msd.models[f]
            end
        ),
        t_next = iszero(updates_output.t_next) ? msd.t_next : rationalize(updates_output.t_next), # If there's no t_next, keep the last one.
    )
end

function step!(mh, t, ommd, rates_fcn, updates_fcn, t_last, msd, solver, monitors, t_end, t_next_suggested)

    # Figure out how big this step can be.

    # Assume the next stop is the next time a user asked for a stop (which might be the end
    # time).
    k_next_requested_stop = findfirst(>(t_last), t)
    if !isnothing(k_next_requested_stop)
        t_next = t[k_next_requested_stop]
    else
        t_next = last(t)
    end

    # But if the integrator suggested a sooner time, take that.
    t_next = min(t_next, t_next_suggested)

    # Ask all of the models what time they want to stop next, and take the soonest.
    t_next = recursively_reduce(msd, t_next) do el, t_next_so_far
        # TODO: Should we use Inf instead of 0 for "whatever t_next is fine with me"?
        if iszero(el.t_next)
            return t_next_so_far
        else
            return min(t_next_so_far, rationalize(el.t_next))
        end
    end

    # Perform the continuous-time update from t_last to t_next.
    # println("Stepping from $t_last to $t_next.")

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

    # Update the monitors.
    for m in monitors
        Monitors.update_monitor!(m, t_next) # TODO: Let these stop the loop.
    end

    return (t_next, msd, UnknownStopReason(), t_next_suggested)

end

function loop!(mh, t, ommd, rates_fcn, updates_fcn, msd, solver, monitors)
    t_completed = first(t)
    t_end = last(t)
    t_next_suggested = t_end
    stop = UnknownStopReason()
    while isa(stop, UnknownStopReason)
        try
            t_completed, msd, stop, t_next_suggested = step!(mh, t, ommd, rates_fcn, updates_fcn, t_completed, msd, solver, monitors, t_end, t_next_suggested)
        catch err
            trace = stacktrace(catch_backtrace())
            showerror(stderr, err, trace)
            stop = EncounteredError(float(t_completed), err, trace)
        end
    end
    return (t_completed, msd, stop)
end

############
# simulate #
############

"""
TODO
"""
function simulate(
    model_prototype;
    t, # Any collection; sim starts at first(t) and goes to last(t) and breaks at everything in between.
    init_fcn, # Turns the prototype into a model description, which can be turned into a model
    rates_fcn,
    updates_fcn,
    close_fcn = (t, model) -> nothing,
    seed = 0,
    options::SimOptions = SimOptions(),
)
    t = [rationalize(el) for el in t]
    t_start = first(t)
    t_end   = last(t)

    # Run the initialization to get the description of the models given the prototype.
    rng = Xoshiro(seed)
    model_description = init_fcn(t_start, model_prototype, rng)

    # Use those descriptions to build up the time histories.
    log, mh = create_log(options.log, model_description, options.time_dimension)

    # Now that the time histories are started, we have no further use of the
    # VariableDescriptions. Strip those out for the "original minimal model description".
    # We'll always keep this original description around for its random-variable functions.
    ommd = strip_fluff_from_model_description(model_description)

    # We can now fill in the draws to have a "model state description".
    msd = draw_wd(t_start, ommd, ommd)

    # Log the initial stuff.
    log_discrete_stuff!(t_start, mh, ommd)

    # From the model state description, we can build the model itself (the single structure
    # that has fields for all of the variables that were described in the original model
    # description).

    # Create the solver.
    solver = create_solver(options.solver, msd)

    # Create the monitors.
    monitors = map(mo -> Monitors.create_monitor(mo, t_start, t_end), options.monitors)

    # Begin the loop.
    t_end, msd, stop = loop!(mh, t, ommd, rates_fcn, updates_fcn, msd, solver, monitors)

    # Close out the models.
    close_fcn(t_end, model(msd))

    # Wrap up all of the history into a single object.
    history = SimHistory(model_description, log, stop)

    # Close the monitors.
    for m in monitors
        Monitors.close_monitor!(m, t_end)
    end

    return (history, t_end, model(msd))

end

end # module SystemsOfSystems
