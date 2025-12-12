"""
TODO
"""
module Solvers

export create_solver, get_initial_time_step, solve

using ..SystemsOfSystems: ModelStateDescription, RatesOutput, AbstractStopReason, UnknownStopReason, model, draw_wc, copy_model_state_description_except
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
    return (x .+ dt .* x_dot)::T # Just to be clear, this shouldn't change the type.
end

function propagate_set(x::T1, dt, x_dot::T2) where {T1, T2}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            propagate_variable(x[f], dt, x_dot[f])
        end
    )
end

function propagate_models(submodels::NamedTuple, dt, rates_output::NamedTuple)

    # A user's RatesOutput's model entry could contain the models in any order. Here, we
    # build a named tuple that matches the order of the original set of submodels. Plus, if
    # an entry is missing, we fill it in with a blank RatesOutput(). This lets us simply
    # `map` below.
    complete_rates_output = NamedTuple{fieldnames(typeof(submodels))}(
        map(fieldnames(typeof(submodels))) do f
            if hasfield(typeof(rates_output), f)
                rates_output[f]
            else
                RatesOutput()
            end
        end
    )

    # Now this is a simple map and doesn't allocate.
    return map((sm, ro) -> propagate(sm, dt, ro), submodels, complete_rates_output)

end

function propagate(msd::ModelStateDescription, dt, rates_output::RatesOutput)
    return copy_model_state_description_except(
        msd;
        continuous_states = propagate_set(msd.continuous_states, dt, rates_output.rates),
        models = propagate_models(msd.models, dt, rates_output.models),
    )
end

# These propagate for a set of derivatives.

function propagate_variable(x::T, gains, x_dot::NTuple{N, T}) where {T, N}
    # println("propagate_variable for $T")
    return (x .+ sum(gains .* x_dot))::T # Just to be clear, this shouldn't change the type.
end

function propagate_set(x::T1, gains, x_dot::Tuple) where {T1}
    # println("propagate_set for $T1 and $(typeof(x_dot))")
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            propagate_variable(x[f], gains, getfield.(x_dot, f))
        end
    )
end

# `submodels` is a named tuple of ModelStateDescriptions.
# `gains` is a tuple of gains.
# `rates_output` is a tuple (one for each gain) of named tuples holding the RatesOutput
# of each of the submodels (for submodels that have such an output).
function propagate_models(submodels::NamedTuple, gains::Tuple, rates_outputs::Tuple)

    # println("propagate_models for $T1")

    complete_rates_outputs = map(rates_outputs) do ro
        NamedTuple{fieldnames(typeof(submodels))}(
            map(fieldnames(typeof(submodels))) do f
                # fieldtype(typeof(rates_outputs), 1)
                if hasfield(typeof(ro), f) # If we have derivatives for this state...
                    # println("We have a RatesOutput for $f")
                    getfield(ro, f) # Get it for all of them.
                else
                    # println("We DON'T have a RatesOutput for $f")
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
    # println("propagate for $T")
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

function solve(ommd, solver::RungeKutta4, t_last, t_next, msd_km1, rates_fcn, t_end)

    # Make the draws for the continuous-time function.
    msd_km1_with_draws = draw_wc(t_last, t_next, ommd, msd_km1)

    # The first derivative is different because it's an output. The rest are ephemeral.
    msd1 = msd_km1_with_draws
    k1 = rates_fcn(t_last, model(msd1))

    # If there's no actual work to do here, skip the calculations.
    if t_last == t_next

        msd_k = msd_km1_with_draws

    else

        dt    = t_next - t_last
        msd2  = propagate(msd1, dt/2, k1)
        k2    = rates_fcn(t_last + dt/2, model(msd2))
        msd3  = propagate(msd1, dt/2, k2)
        k3    = rates_fcn(t_last + dt, model(msd3))
        msd4  = propagate(msd1, dt, k3)
        k4    = rates_fcn(t_last + dt, model(msd4))

        # This seems more efficient:
        # propagate(
        #     msd_km1_with_draws,
        #     (dt/6, dt/3, dt/3, dt/6),
        #     (k1, k2, k3, k4),
        # )

        # But this doesn't allocate and is actually slightly faster.
        msd_k = msd_km1_with_draws
        msd_k = propagate(msd_k, dt/6, k1)
        msd_k = propagate(msd_k, dt/3, k2)
        msd_k = propagate(msd_k, dt/3, k3)
        msd_k = propagate(msd_k, dt/6, k4)

    end

    return SolverOutputs(;
        t_completed = t_next,
        msd_km1 = msd_km1_with_draws,
        msd_k,
        rates = k1,
        stop = UnknownStopReason(),
        t_next_suggested = t_next + solver.options.dt,
    )

end

###################
# DormandPrince54 #
###################

# TODO: We could let each model provide its own error function, with its own absolute and
# relative tolerances.

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

    table = (   # Butcher tableau (Dormand-Prince 5(4) by default)
        (1/5, 1/5),                 # c_2, a_2,1
        (3/10, 3/40, 9/40),         # c_3, a_3,1 a_3,2
        (4/5, 44/45, -56/15, 32/9), # etc.
        (8/9, 19372/6561, −25360/2187, 64448/6561, −212/729),
        (1., 9017/3168, −355/33, 46732/5247, 49/176, −5103/18656),
        (1., 35/384, 0., 500/1113, 125/192, −2187/6784, 11/84),
        (35/384, 0., 500/1113, 125/192, −2187/6784, 11/84, 0.), # The first-same-as-last property is useless here due to the discrete update.
        (5179/57600, 0., 7571/16695, 393/640, −92097/339200, 187/2100, 1/40),
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

        dt = t_next - t_last

        # Make the draws for the continuous-time function.
        msd_km1_with_draws = draw_wc(t_last, t_next, ommd, msd_km1)

        # We do the first step whether we're stopping on this sample or not.
        msd1 = msd_km1_with_draws
        k1 = rates_fcn(t_last, model(msd1))

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
                ki = rates_fcn(t_last + dt * ci, model(msdi))
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
                t_next_suggested = t_completed + dt_suggested
                # println("That step worked. t_next_suggested = $(float(t_next_suggested)).")
                break

            else

                # println("Stepping from $(float(t_last)) to $(float(t_next)) produced too much error.")
                # @show max_normalized_error
                t_next = t_last + dt_suggested
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
        t_completed = rationalize(t_completed),
        msd_km1 = msd_km1_with_draws,
        msd_k = msd_k,
        rates = k1,
        stop = stop,
        t_next_suggested = rationalize(t_next_suggested),
    )

end

end
