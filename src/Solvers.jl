"""
TODO
"""
module Solvers

export create_solver, get_initial_time_step, solve

using ..SystemsOfSystems: ModelStateDescription, RatesOutput, AbstractStopReason, UnknownStopReason, model, draw_wc, draw_wc!, copy_model_state_description_except
import SystemsOfSystems

##################
# AbstractSolver #
##################

abstract type AbstractSolverOptions end
abstract type AbstractSolver end

# This is what the "solve" method is expected to output.
@kwdef struct SolverOutputs{T1 <: ModelStateDescription, T2 <: RatesOutput}
    t_completed::Rational{Int64}
    msd_km1::T1
    msd_k::T1
    rates::T2
    stop::AbstractStopReason
    t_next_suggested::Rational{Int64}
end

# Adaptive solvers will need to say when solving just isn't working.
struct SolverFailedToConverge <: AbstractStopReason
    time::Float64
end
SystemsOfSystems.describe(stop::SolverFailedToConverge) = "The solver failed to converge at time $(float(stop.time))."

###########
# Helpers #
###########

# These propagate for a single derivative.

function propagate_variable(x::T, dt, x_dot::T) where {T}
    return (x + dt * x_dot)::T # Just to be clear, this shouldn't change the type.
end

function propagate_set(x::T1, dt, x_dot::T2) where {T1, T2}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(typeof(x_dot), f)
                propagate_variable(x[f], dt, x_dot[f])
            else
                x[f]
            end
        end
    )
end

function propagate_submodels!(submodels::NamedTuple, dt, rates_output::NamedTuple)
    foreach(fieldnames(typeof(rates_output))) do f
        propagate_msd!(submodels[f], dt, rates_output[f])
    end
    return nothing
end

function propagate_msd!(msd::ModelStateDescription, dt, rates_output::RatesOutput)
    msd.continuous_states = propagate_set(msd.continuous_states, dt, rates_output.rates)
    propagate_submodels!(msd.models, dt, rates_output.models)
    return nothing
end

function copy_xc_to_model!(m, xc)
    foreach(fieldnames(typeof(xc))) do f
    # for f in fieldnames(typeof(xc))
        setfield!(m, f, xc[f])
    end
    return nothing
end

function propagate_to_submodels!(m, models_msd, models_rates_output, dt)
    foreach(fieldnames(typeof(models_rates_output))) do f
        propagate_to_model!(getproperty(m, f), models_msd[f], dt, models_rates_output[f])
    end
end

function propagate_to_model!(m, msd, dt, rates_output)
    xc = propagate_set(msd.continuous_states, dt, rates_output.rates)
    copy_xc_to_model!(m, xc)
    # for f in fieldnames(typeof(msd.models))
    #     if hasfield(typeof(rates_output.models), f)
    #         propagate_to_model!(getproperty(m, f), msd.models[f], dt, rates_output.models[f])
    #     end
    # end
    propagate_to_submodels!(m, msd.models, rates_output.models, dt)
    return nothing
end

function copy_continuous_state_for_submodels!(m, models_msd)
    foreach(fieldnames(typeof(models_msd))) do f
        copy_continuous_state!(getproperty(m, f), models_msd[f])
    end
end

function copy_continuous_state!(m, msd)
    copy_xc_to_model!(m, msd.continuous_states)
    # for f in fieldnames(typeof(msd.models))
    #     copy_continuous_state!(getproperty(m, f), msd.models[f])
    # end
    copy_continuous_state_for_submodels!(m, msd.models)
    return nothing
end

# These propagate for a set of derivatives.

function propagate_variable(x::T, gains, x_dot::NTuple{N, T}) where {T, N}
    # return (x .+ sum(gains .* x_dot))::T # Just to be clear, this shouldn't change the type.
    return (x + sum(gains .* x_dot))::T # Just to be clear, this shouldn't change the type.
end

function propagate_set(x::T1, gains, x_dot::Tuple) where {T1}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(typeof(first(x_dot)), f) # TODO: Check this for efficiency.
                propagate_variable(x[f], gains, getfield.(x_dot, f))
            else
                x[f] # Allow fields to not be updated (empty rates output).
            end
        end
    )
end

# `submodels` is a named tuple of ModelStateDescriptions.
# `gains` is a tuple of gains.
# `rates_output` is a tuple (one for each gain) of named tuples holding the RatesOutput
# of each of the submodels (for submodels that have such an output).
function propagate_models(submodels::NamedTuple, gains::Tuple, rates_outputs::Tuple)
    complete_rates_outputs = map(rates_outputs) do ro
        NamedTuple{fieldnames(typeof(submodels))}(
            map(fieldnames(typeof(submodels))) do f
                if hasfield(typeof(ro), f) # If we have derivatives for this state...
                    getproperty(ro, f) # Get it for all of them.
                else
                    RatesOutput()
                end
            end
        )
    end
    return map(
        (sm, ro...) -> propagate(sm, gains, ro),
        submodels, complete_rates_outputs...
    )
end

function propagate(msd::ModelStateDescription{T}, gains::Tuple, rates_outputs::Tuple) where {T}
    return copy_model_state_description_except(
        msd;
        continuous_states = propagate_set(msd.continuous_states, gains, getfield.(rates_outputs, :rates)),
        models = propagate_models(msd.models, gains, getfield.(rates_outputs, :models)),
    )
end

###############
# RungeKutta4 #
###############

"""
TODO
"""
struct RungeKutta4Options <: AbstractSolverOptions
    dt::Rational{Int64}
end
RungeKutta4Options(; dt, ) = RungeKutta4Options(rationalize(dt))
struct RungeKutta4 <: AbstractSolver
    options::RungeKutta4Options
end
create_solver(options::RungeKutta4Options, msd::ModelStateDescription) = RungeKutta4(options)

get_initial_time_step(solver::RungeKutta4) = solver.options.dt

# TODO: It seems like there's a lot about `solve` that could be abstracted and simplified.
function solve(ommd, solver::RungeKutta4, t_last, t_next, m, msd_km1, rates_fcn, t_end)

    t_last_f = float(t_last)
    t_next_f = float(t_next)

    # Make the continuous-time draws, storing the results directly in the model form.
    draw_wc!(m, t_last_f, t_next_f, ommd)

    # The first derivative is different because it's an output. The rest are ephemeral.
    k1 = rates_fcn(t_last_f, m)

    # If there's no actual work to do here, skip the calculations.
    if t_last == t_next

        msd_k = msd_km1 # TODO: Do I need a deepcopy here?

    else

        # Propagate from the continuous-time state in msd_km1 using the k1 rate, storing the
        # results directly in the model.
        dt = t_next_f - t_last_f
        propagate_to_model!(m, msd_km1, dt/2, k1) # Propagates directly into the model
        k2 = rates_fcn(t_last_f + dt/2, m)
        propagate_to_model!(m, msd_km1, dt/2, k2)
        k3 = rates_fcn(t_last_f + dt/2, m)
        propagate_to_model!(m, msd_km1, dt, k3)
        k4 = rates_fcn(t_last_f + dt, m)

        # Update the model state description with the propagated continuous-time states.
        # msd_k = deepcopy(msd_km1) # TODO: There's no real reason to do this. We could just commit to updating the single MSD.
        msd_k = msd_km1 # TODO: This is wrong. It was just a timing test.
        propagate_msd!(msd_k, dt/6, k1)
        propagate_msd!(msd_k, dt/3, k2)
        propagate_msd!(msd_k, dt/3, k3)
        propagate_msd!(msd_k, dt/6, k4)

        # Now copy that to the model itself.
        copy_continuous_state!(m, msd_k)

    end

    return SolverOutputs(;
        t_completed = t_next, # This should already be a rational.
        msd_km1 = msd_km1,
        msd_k,
        rates = k1,
        stop = UnknownStopReason(),
        t_next_suggested = t_next + solver.options.dt, # Already rational
    )

end

###################
# DormandPrince54 #
###################

"""
TODO
"""
struct DormandPrince54Options <: AbstractSolverOptions
    initial_dt::Rational{Int64}
    max_dt::Rational{Int64}
    abs_tol::Float64
    rel_tol::Float64
end
DormandPrince54Options(;
    initial_dt = 1//1,
    max_dt = 1//0,
    abs_tol = 1e-3, # TODO: Figure out what's most common for these.
    rel_tol = 1e-5,
) = DormandPrince54Options(
    rationalize(initial_dt),
    rationalize(max_dt),
    abs_tol,
    rel_tol,
)
struct DormandPrince54 <: AbstractSolver
    options::DormandPrince54Options
    # TODO: Tables and types
end
create_solver(options::DormandPrince54Options, msd::ModelStateDescription) = DormandPrince54(options)

get_initial_time_step(solver::DormandPrince54) = solver.options.initial_dt

# This returns how much of the allowable error tolerance was "used" by this intergration
# step, reporting only the worst case (largest fraction of tolerance used).
function get_max_normalized_error(solver, msd1, msd2, max_so_far)
    if !isempty(msd1.continuous_states)
        max_here = maximum( # max from each variable
            maximum( # max over each element of the variable
                # For each element, we'll use the more permissive of the absolute and
                # relative tolerances. If the relative tolerance is super small (or maybe
                # actually zero if x is 0), then we'll normalize by the absolute tolerance.
                # If the relatively tolerance is big (because x is big), then we'll
                # normalize by the relative tolerance.
                if solver.options.abs_tol > abs(x) * solver.options.rel_tol # abs_tol yields largest step
                    abs(dx) / solver.options.abs_tol
                else
                    abs(dx/x) / solver.options.rel_tol # Clearly, there is no divide-by-zero here.
                end
                for (x, dx) in zip(x1, (x1 - x2))
            )
            for (x1, x2) in zip(msd1.continuous_states, msd2.continuous_states)
        )
        max_so_far = max(max_so_far, max_here)
    end
    for (m1, m2) in zip(msd1.models, msd2.models)
        max_so_far = get_max_normalized_error(solver, m1, m2, max_so_far)
    end
    return max_so_far
end

function solve(ommd, solver::DormandPrince54, t_last, t_next, msd_km1, rates_fcn, t_end)

    t_last_f = float(t_last)
    t_next_f = float(t_next)

    table = (   # Butcher tableau (Dormand-Prince 5(4) by default)
        (1/5, 1/5),                 # c_2, a_2,1
        (3/10, 3/40, 9/40),         # c_3, a_3,1 a_3,2
        (4/5, 44/45, -56/15, 32/9), # etc.
        (8/9, 19372/6561, -25360/2187, 64448/6561, -212/729),
        (1., 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
        (1., 35/384, 0., 500/1113, 125/192, -2187/6784, 11/84),
        (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.), # The first-same-as-last property is useless here due to the discrete update.
        (5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40),
    )

    # These will all get updated in the loop.
    stop = UnknownStopReason()
    msd_km1_with_draws = msd_km1
    msd_k = msd_km1
    k1 = nothing
    t_completed = t_last
    t_next_suggested = t_next + solver.options.max_dt # Placeholder

    # Make sure we don't take too many steps.
    n_allowable_failed_steps = 20
    n_failed_steps = 0

    while true

        # println("continuous_step! from $(float(t_last)) to $(float(t_next))")

        dt = t_next_f - t_last_f

        # Make the draws for the continuous-time function.
        msd_km1_with_draws = draw_wc(t_last_f, t_next_f, ommd, msd_km1)

        # We do the first step whether we're stopping on this sample or not.
        msd1 = msd_km1_with_draws
        k1 = rates_fcn(t_last_f, model(msd1))

        # See if it's time to stop.
        if t_last == t_end

            msd_k = msd_km1_with_draws
            break

        else

            # TODO: This is inefficient. See what we can redo here.
            ks = (k1,)
            for i in 1:length(table) - 2
                ci = table[i][1]
                as = table[i][2:end]
                msdi = propagate(msd_km1_with_draws, dt .* as, ks)
                ki = rates_fcn(t_last_f + dt * ci, model(msdi))
                ks = (ks..., ki) # TODO: This is a particularly silly pattern.
            end

            # Assemble the derivatives into the update.
            bs = table[end-1]
            b_hats = table[end]
            msd_k = propagate(msd_km1_with_draws, dt .* bs, ks)
            msd_k_hat = propagate(msd_km1_with_draws, dt .* b_hats, ks)

            # Figure out the error between the two different solutions. Here, we'll use the
            # "normalized" error, where the error is normalized by its tolerance, which may
            # be either an absolute or relative tolerance.
            max_normalized_error = get_max_normalized_error(solver, msd_k, msd_k_hat, 0.)

            # Choose the next time step.
            p = 4 # Get this from the same place as the table.
            σ = 0.8 # Safety factor
            dt_suggested = σ * dt * max_normalized_error^(-1/(p+1))
            # dt_suggested = min(2 * dt, dt_suggested) # Never grow the step by more than this factor.
            # dt_suggested = max(dt / 3, dt_suggested) # Never shrink the step by more than this factor.

            # If no error was above its tolerance...
            if max_normalized_error < 1.

                # Accept the update.
                t_completed = t_next
                t_next_suggested = rationalize(t_next_f + dt_suggested)
                # println("That step worked. t_next_suggested = $(float(t_next_suggested)).")
                break

            else

                # println("Stepping from $(float(t_last)) to $(float(t_next)) produced too much error.")
                # @show max_normalized_error
                t_next_f = t_last_f + dt_suggested
                t_next = rationalize(t_next_f)
                # println("Trying again with t_next = $(float(t_next)).")
                n_failed_steps += 1
                if n_failed_steps == n_allowable_failed_steps
                    stop = SolverFailedToConverge(t_last)
                    msd_k = msd_km1
                    break
                end

            end

        end

    end

    # Reported how much of the intended step we completed, the updated model state
    # description, and the suggested next step's end time.
    return SolverOutputs(;
        t_completed = t_completed,
        msd_km1 = msd_km1_with_draws,
        msd_k = msd_k,
        rates = k1,
        stop = stop,
        t_next_suggested = t_next_suggested,
    )

end

end
