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
    return (x + dt * x_dot)::T # Just to be clear, this shouldn't change the type.
end

@generated function propagate_set(x::T1, dt, x_dot::T2) where {T1, T2}
    names = fieldnames(T1)
    values = [
        hasfield(T2, name) ?
            :(propagate_variable(getfield(x, $(QuoteNode(name))), dt, getfield(x_dot, $(QuoteNode(name))))) :
            :(getfield(x, $(QuoteNode(name))))
        for name in names
    ]
    return :(NamedTuple{$names}(($(values...),)))
end

is_empty_rates_output(rates_output::RatesOutput) =
    isempty(fieldnames(typeof(rates_output.rates))) &&
    isempty(fieldnames(typeof(rates_output.models)))

@generated function propagate_models(
    submodels::NamedTuple{names}, dt, rates_output::TRO,
) where {names, TRO}
    values = [
        hasfield(TRO, name) ?
            :(propagate(getfield(submodels, $(QuoteNode(name))), dt, getfield(rates_output, $(QuoteNode(name))))) :
            :(getfield(submodels, $(QuoteNode(name))))
        for name in names
    ]
    return :(NamedTuple{$names}(($(values...),)))
end

function propagate(msd::ModelStateDescription, dt, rates_output::RatesOutput)
    is_empty_rates_output(rates_output) && return msd
    return copy_model_state_description_except(
        msd;
        continuous_states = propagate_set(msd.continuous_states, dt, rates_output.rates),
        models = propagate_models(msd.models, dt, rates_output.models),
    )
end

function propagate_variable_rk4(x::T, dt, k1::T, k2::T, k3::T, k4::T) where {T}
    return (x + dt/6 * (k1 + 2*k2 + 2*k3 + k4))::T
end

function propagate_set_rk4(x::T1, dt, r1, r2, r3, r4) where {T1}
    return NamedTuple{fieldnames(T1)}(
        map(fieldnames(T1)) do f
            if hasfield(typeof(r1), f)
                propagate_variable_rk4(x[f], dt, r1[f], r2[f], r3[f], r4[f])
            else
                x[f]
            end
        end
    )
end

function propagate_models_rk4(submodels::NamedTuple, dt, m1::NamedTuple, m2::NamedTuple, m3::NamedTuple, m4::NamedTuple)
    names = fieldnames(typeof(submodels))
    complete_m1 = NamedTuple{names}(map(names) do f
        hasfield(typeof(m1), f) ? getfield(m1, f) : RatesOutput()
    end)
    complete_m2 = NamedTuple{names}(map(names) do f
        hasfield(typeof(m2), f) ? getfield(m2, f) : RatesOutput()
    end)
    complete_m3 = NamedTuple{names}(map(names) do f
        hasfield(typeof(m3), f) ? getfield(m3, f) : RatesOutput()
    end)
    complete_m4 = NamedTuple{names}(map(names) do f
        hasfield(typeof(m4), f) ? getfield(m4, f) : RatesOutput()
    end)
    return map(
        (sm, ro1, ro2, ro3, ro4) -> propagate_rk4(sm, dt, ro1, ro2, ro3, ro4),
        submodels, complete_m1, complete_m2, complete_m3, complete_m4,
    )
end

function propagate_rk4(msd::ModelStateDescription, dt, k1::RatesOutput, k2::RatesOutput, k3::RatesOutput, k4::RatesOutput)
    is_empty_rates_output(k1) && is_empty_rates_output(k2) &&
        is_empty_rates_output(k3) && is_empty_rates_output(k4) && return msd
    return copy_model_state_description_except(
        msd;
        continuous_states = propagate_set_rk4(msd.continuous_states, dt, k1.rates, k2.rates, k3.rates, k4.rates),
        models = propagate_models_rk4(msd.models, dt, k1.models, k2.models, k3.models, k4.models),
    )
end

# These propagate for a set of derivatives.

function propagate_variable(x::T, gains, x_dot::NTuple{N, T}) where {T, N}
    # return (x .+ sum(gains .* x_dot))::T # Just to be clear, this shouldn't change the type.
    return (x + sum(gains .* x_dot))::T # Just to be clear, this shouldn't change the type.
end

@generated function propagate_set(x::T1, gains, x_dot::TX) where {T1, TX <: Tuple}
    names = fieldnames(T1)
    n = fieldcount(TX)
    first_rates_type = fieldtype(TX, 1)
    values = [
        hasfield(first_rates_type, name) ?
            :(propagate_variable(
                getfield(x, $(QuoteNode(name))), gains,
                ntuple(i -> getfield(getfield(x_dot, i), $(QuoteNode(name))), Val($n)),
            )) :
            :(getfield(x, $(QuoteNode(name))))
        for name in names
    ]
    return :(NamedTuple{$names}(($(values...),)))
end

# `submodels` is a named tuple of ModelStateDescriptions.
# `gains` is a tuple of gains.
# `rates_output` is a tuple (one for each gain) of named tuples holding the RatesOutput
# of each of the submodels (for submodels that have such an output).
@generated function propagate_models(
    submodels::NamedTuple{names}, gains::Tuple, rates_outputs::TRO,
) where {names, TRO <: Tuple}
    n = fieldcount(TRO)
    first_models_type = fieldtype(TRO, 1)
    values = [
        hasfield(first_models_type, name) ?
            :(propagate(
                getfield(submodels, $(QuoteNode(name))), gains,
                ntuple(i -> getfield(getfield(rates_outputs, i), $(QuoteNode(name))), Val($n)),
            )) :
            :(getfield(submodels, $(QuoteNode(name))))
        for name in names
    ]
    return :(NamedTuple{$names}(($(values...),)))
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
handles_internal_substepping(::AbstractSolver) = false
handles_internal_substepping(::RungeKutta4) = true

# TODO: It seems like there's a lot about `solve` that could be abstracted and simplified.
function solve(ommd, solver::RungeKutta4, t_last, t_next, msd_km1, rates_fcn, t_end)
    dt_solver = solver.options.dt
    function rk4_substep(msd_start, t_start, t_stop, msd_with_draws, k1)
        # If there's no actual work to do here, skip the calculations.
        if t_start == t_stop
            return msd_with_draws
        else
            t_start_f = float(t_start)
            t_stop_f = float(t_stop)
            dt = t_stop_f - t_start_f
            msd2 = propagate(msd_with_draws, dt/2, k1)
            k2 = rates_fcn(t_start_f + dt/2, model(msd2))
            msd3 = propagate(msd_with_draws, dt/2, k2)
            k3 = rates_fcn(t_start_f + dt/2, model(msd3))
            msd4 = propagate(msd_with_draws, dt, k3)
            k4 = rates_fcn(t_start_f + dt, model(msd4))
            msd_stop = msd_with_draws
            msd_stop = propagate(msd_stop, dt/6, k1)
            msd_stop = propagate(msd_stop, dt/3, k2)
            msd_stop = propagate(msd_stop, dt/3, k3)
            msd_stop = propagate(msd_stop, dt/6, k4)
            return msd_stop
        end
    end

    t_sub_last = t_last
    t_sub_next = min(t_next, t_sub_last + dt_solver)
    first_msd = draw_wc(float(t_sub_last), float(t_sub_next), ommd, msd_km1)
    first_rates = rates_fcn(float(t_sub_last), model(first_msd))
    msd_k = rk4_substep(msd_km1, t_sub_last, t_sub_next, first_msd, first_rates)

    while t_sub_next != t_next
        t_sub_last = t_sub_next
        t_sub_next = min(t_next, t_sub_last + dt_solver)
        msd_with_draws = draw_wc(float(t_sub_last), float(t_sub_next), ommd, msd_k)
        k1 = rates_fcn(float(t_sub_last), model(msd_with_draws))
        msd_k = rk4_substep(msd_k, t_sub_last, t_sub_next, msd_with_draws, k1)
    end

    return SolverOutputs(;
        t_completed = t_next, # This should already be a rational.
        msd_km1 = first_msd,
        msd_k,
        rates = first_rates,
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

get_initial_time_step(solver::DormandPrince54) = min(solver.options.initial_dt, solver.options.max_dt)

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
    t_next_suggested = t_next + solver.options.max_dt

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

                # If the suggested step size is less than the maximum, use it. Otherwise,
                # stick with the maximum.
                t_completed = t_next
                if dt_suggested < solver.options.max_dt
                    t_next_suggested = rationalize(t_next_f + dt_suggested)
                else
                    t_next_suggested = t_next + solver.options.max_dt
                end
                # println("That step worked. t_next_suggested = $(float(t_next_suggested)).")
                break

            else

                # println("Stepping from $(float(t_last)) to $(float(t_next)) produced too much error.")
                # @show max_normalized_error
                t_next_f = t_last_f + min(dt_suggested, float(solver.options.max_dt))
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
