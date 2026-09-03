"""
The `Solvers` module contains continuous-time integration algorithms and their step-size
controllers.

The simulation loop and numerical solver deliberately meet at a narrow boundary: one call
to `step!` attempts and returns exactly one accepted numerical step without crossing a
rational time bound. Every accepted step remains visible to the simulation loop, which then
logs the sample, runs hooks, draws discrete random variables, and performs discrete updates.

Numerical methods in this module do not know how SystemsOfSystems stores model state. They
use the operations supplied by `ContinuousProblems` to prepare a random interval, evaluate
rates, form linear combinations of derivatives, and measure normalized error. This
separation makes a Butcher tableau a description of mathematics rather than a second
implementation of the simulation engine.
"""
module Solvers

public AbstractSolverOptions, AbstractIntegrator, AbstractSolver,
    StepRequest, AcceptedStep, SolverFailure,
    SolverFailedToConverge, SolverStepSizeUnderflow,
    Ralston2Options, RungeKutta4Options, DormandPrince54Options,
    draw_continuous_random_variables, create_integrator, step!

using ..ContinuousProblems: ContinuousProblem, evaluate_rates, normalized_error,
    draw_continuous_random_variables, propagate
using ..SimulationTimes: ExactTime, exact_time, float_duration, solver_time, time_isless
using ..SystemsOfSystems: AbstractFailureReason, ModelStateDescription
import ..SystemsOfSystems

############################
# Solver Boundary and Types #
############################

"""
The common supertype for immutable, user-facing solver configuration.

Options may be reused across simulations. Runtime state such as the next adaptive step size
belongs to an `AbstractIntegrator` created from the options for one simulation.
"""
abstract type AbstractSolverOptions end

"""
The common supertype for a runtime continuous-time integrator.

An integrator may retain controller history and numerical caches. It is owned by one
simulation and receives the potentially discontinuously updated model state on every call to
`step!`.
"""
abstract type AbstractIntegrator end

# Keep the old abstract name available while downstream code transitions to the clearer
# distinction between reusable solver options and a per-simulation integrator.
const AbstractSolver = AbstractIntegrator

"""
    StepRequest(t_start, t_bound, state, t_next_crv_draw)

A container for one accepted continuous-time step request beginning at the official
rational time `t_start`. The integrator may choose any rational endpoint no later than
`t_bound`.

`t_bound` is selected by the simulation scheduler from user-requested times,
model-requested times, and the overall end time. It is a hard boundary: no numerical stage
may cause the accepted state to be labeled with a time beyond it.

`t_next_crv_draw` is the endpoint of the currently committed continuous-random interval.
Equality with `t_start` means the solver should begin a new interval. Otherwise, the solver
holds the current draws until it reaches that endpoint.
"""
struct StepRequest{T <: Rational, S <: ModelStateDescription}
    t_start::T
    t_bound::T
    state::S
    t_next_crv_draw::T
end

"""
    AcceptedStep

The result of exactly one accepted numerical step.

`state_at_start` includes the continuous random draws belonging to the accepted interval,
and `rates_at_start` is the authoritative rates evaluation for that accepted sample. The
simulation loop logs those values and considers their model stop requests. Intermediate
stage outputs and stop requests never cross this boundary.

`next_dt` is a floating-point controller suggestion. It is deliberately a duration rather
than an absolute time; the scheduler converts it into an official rational endpoint for the
next attempt.

`t_next_crv_draw` carries the committed continuous-random interval endpoint into the next
step request.
"""
struct AcceptedStep{
    T <: Rational,
    S <: ModelStateDescription,
    R,
}
    t_end::T
    state_at_start::S
    rates_at_start::R
    state_at_end::S
    next_dt::Float64
    t_next_crv_draw::T
end

"""
    SolverFailure(time, reason)

A result indicating that no acceptable numerical step could be produced from `time`.

A solver failure is not a model stop request. Keeping it as a distinct result prevents the
normal accepted-step path from carrying an abstract stop field and makes it impossible for
the simulation loop to run hooks or updates for a step that was never accepted.
"""
struct SolverFailure{T <: Rational, R <: AbstractFailureReason}
    time::T
    reason::R
end

"""
The adaptive solver exhausted its allowed rejected attempts without satisfying tolerance.
"""
struct SolverFailedToConverge <: AbstractFailureReason
    time::Float64
end

function SystemsOfSystems.describe(failure::SolverFailedToConverge)
    return "The solver failed to converge at time $(failure.time)."
end

"""
The proposed floating-point step was too small to produce a later official rational time.

Reporting this explicitly is preferable to repeatedly accepting a zero-duration step, which
would leave the simulation loop unable to make progress.
"""
struct SolverStepSizeUnderflow <: AbstractFailureReason
    time::Float64
    proposed_dt::Float64
end

function SystemsOfSystems.describe(failure::SolverStepSizeUnderflow)
    return "The proposed solver step of $(failure.proposed_dt) did not advance time at " *
        "$(failure.time)."
end

####################
# Butcher Tableaus #
####################

"""
    ExplicitRungeKuttaTableau(a, b, c; embedded_b, order, embedded_order)

An explicit Runge-Kutta method expressed as a Butcher tableau.

The first stage is always evaluated at the beginning state, so `a` and `c` describe stages
two through `N`. `a[i]` therefore contains exactly `i` weights, while `c[i]` gives the time
fraction for that stage. `b` contains all `N` weights used to form the primary solution.
`embedded_b` is either `nothing` or another `N`-tuple used to estimate local error.

Tuple sizes are part of the concrete tableau type. This lets the recursive stage evaluator
specialize each stage without growing a tuple inside a runtime loop. A future public
constructor for arbitrary tableaus can validate and convert user data into this same static
representation during integrator initialization.
"""
struct ExplicitRungeKuttaTableau{A, B, C, EB}
    a::A
    b::B
    c::C
    embedded_b::EB
    order::Int
    embedded_order::Int
end

function ExplicitRungeKuttaTableau(
    a::Tuple,
    b::Tuple,
    c::Tuple;
    embedded_b = nothing,
    order,
    embedded_order = 0,
)

    # Validate the structural relationships once. Numerical stepping can then assume the
    # tableau is coherent and devote its inner loop entirely to stage evaluation.
    n_stages = length(b)
    length(a) == n_stages - 1 || throw(ArgumentError(
        "An explicit $n_stages-stage tableau requires $(n_stages - 1) rows in a.",
    ))
    length(c) == n_stages - 1 || throw(ArgumentError(
        "An explicit $n_stages-stage tableau requires $(n_stages - 1) entries in c.",
    ))
    for index in eachindex(a)
        length(a[index]) == index || throw(ArgumentError(
            "Row $index of a must contain $index coefficients.",
        ))
    end
    if !isnothing(embedded_b)
        length(embedded_b) == n_stages || throw(ArgumentError(
            "The embedded weights must contain $n_stages coefficients.",
        ))
        embedded_order > 0 || throw(ArgumentError(
            "An embedded tableau must provide a positive embedded_order.",
        ))
    end

    return ExplicitRungeKuttaTableau(
        a,
        b,
        c,
        embedded_b,
        Int(order),
        Int(embedded_order),
    )

end

# The built-in methods are immutable constants. Their tuple shapes and coefficient types are
# visible to the compiler wherever the corresponding integrator is specialized.

const RALSTON_2_TABLEAU = ExplicitRungeKuttaTableau(
    (
        (2/3,),
    ),
    (1/4, 3/4),
    (2/3,);
    order = 2,
)

const RUNGE_KUTTA_4_TABLEAU = ExplicitRungeKuttaTableau(
    (
        (1/2,),
        (0., 1/2),
        (0., 0., 1.),
    ),
    (1/6, 1/3, 1/3, 1/6),
    (1/2, 1/2, 1.);
    order = 4,
)

const DORMAND_PRINCE_54_TABLEAU = ExplicitRungeKuttaTableau(
    (
        (1/5,),
        (3/40, 9/40),
        (44/45, -56/15, 32/9),
        (19372/6561, -25360/2187, 64448/6561, -212/729),
        (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
        (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84),
    ),
    (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.),
    (1/5, 3/10, 4/5, 8/9, 1., 1.);
    embedded_b = (
        5179/57600,
        0.,
        7571/16695,
        393/640,
        -92097/339200,
        187/2100,
        1/40,
    ),
    order = 5,
    embedded_order = 4,
)

#########################
# Step-size Controllers #
#########################

"""
A step-size controller that uses the same floating-point duration after every accepted
step.
"""
struct FixedStepController
    dt::Float64
end

"""
An adaptive step-size controller for an embedded Runge-Kutta method using a scalar
normalized local-error estimate.

The controller is mutable because `next_dt` is runtime history belonging to one simulation.
The remaining fields define policy: maximum duration, absolute and relative tolerances,
safety factor, and the number of rejected attempts allowed for one accepted step.
"""
mutable struct EmbeddedAdaptiveController
    next_dt::Float64
    max_dt::Float64
    absolute_tolerance::Float64
    relative_tolerance::Float64
    safety_factor::Float64
    max_rejections::Int
end

"""
A runtime explicit Runge-Kutta integrator composed from a tableau and a controller.

The same numerical kernel serves fixed and adaptive methods. Their behavior differs only in
how the controller chooses an endpoint and whether the tableau supplies embedded weights.
"""
struct RungeKuttaIntegrator{M, C} <: AbstractIntegrator
    method::M
    controller::C
end

#######################
# User-facing Options #
#######################

"""
    Ralston2Options(; dt)

A container for the second-order Ralston Runge-Kutta solver options, where `dt` is the
requested fixed official step spacing in seconds. Scheduled, model-requested, and
user-requested times remain hard bounds and may shorten an individual step.
"""
struct Ralston2Options <: AbstractSolverOptions
    dt::ExactTime
end

function Ralston2Options(; dt)
    rational_dt = exact_time(dt)
    rational_dt > 0 || throw(ArgumentError("dt must be positive."))
    return Ralston2Options(rational_dt)
end

"""
    RungeKutta4Options(; dt)

A container for the classical fourth-order Runge-Kutta solver options, where `dt` is the
fixed official step spacing. Scheduled user and model times remain hard bounds and may
shorten an individual step.
"""
struct RungeKutta4Options <: AbstractSolverOptions
    dt::ExactTime
end

function RungeKutta4Options(; dt)
    rational_dt = exact_time(dt)
    rational_dt > 0 || throw(ArgumentError("dt must be positive."))
    return RungeKutta4Options(rational_dt)
end

"""
    DormandPrince54Options(; initial_dt, max_dt, abs_tol, rel_tol)

A container for the embedded Dormand-Prince 5(4) solver options. The fifth-order solution
advances the model, while the fourth-order solution estimates local error for adaptive step
control.
"""
struct DormandPrince54Options <: AbstractSolverOptions
    initial_dt::ExactTime
    max_dt::ExactTime
    abs_tol::Float64
    rel_tol::Float64
end

function DormandPrince54Options(;
    initial_dt = 1//1,
    max_dt = 1//0,
    abs_tol = 1e-3,
    rel_tol = 1e-5,
)

    rational_initial_dt = exact_time(initial_dt)
    rational_max_dt = exact_time(max_dt)
    rational_initial_dt > 0 || throw(ArgumentError("initial_dt must be positive."))
    rational_max_dt > 0 || throw(ArgumentError("max_dt must be positive."))
    abs_tol > 0 || throw(ArgumentError("abs_tol must be positive."))
    rel_tol >= 0 || throw(ArgumentError("rel_tol cannot be negative."))

    return DormandPrince54Options(
        rational_initial_dt,
        rational_max_dt,
        abs_tol,
        rel_tol,
    )

end

"""
    create_integrator(options, problem, initial_state)

Creates runtime solver state for one simulation. `problem` and `initial_state` are accepted
by the interface even when a particular method does not yet require initialization caches.
"""
function create_integrator(
    options::Ralston2Options,
    problem::ContinuousProblem,
    initial_state::ModelStateDescription,
)
    return RungeKuttaIntegrator(
        RALSTON_2_TABLEAU,
        FixedStepController(float(options.dt)),
    )
end

function create_integrator(
    options::RungeKutta4Options,
    problem::ContinuousProblem,
    initial_state::ModelStateDescription,
)
    return RungeKuttaIntegrator(
        RUNGE_KUTTA_4_TABLEAU,
        FixedStepController(float(options.dt)),
    )
end

function create_integrator(
    options::DormandPrince54Options,
    problem::ContinuousProblem,
    initial_state::ModelStateDescription,
)
    return RungeKuttaIntegrator(
        DORMAND_PRINCE_54_TABLEAU,
        EmbeddedAdaptiveController(
            min(float(options.initial_dt), float(options.max_dt)),
            float(options.max_dt),
            options.abs_tol,
            options.rel_tol,
            0.8,
            20,
        ),
    )
end

#############################
# Explicit Runge-Kutta Core #
#############################

"""
Converts a controller's floating-point duration into one official numerical interval.

The returned endpoint is rational and never crosses `t_bound`. A hard-bound interval derives
its numerical duration from the exact rational event times. A soft solver-selected interval
uses the difference between its floating-point endpoint labels, preserving the behavior of
the floating-point adaptive controller rather than treating an approximate rational label as
an exact numerical duration.
"""
function choose_step_interval(t_start::Rational, t_bound::Rational, proposed_dt::Float64)

    if !(proposed_dt > 0.)
        return nothing
    end
    if !time_isless(t_start, t_bound)
        return nothing
    end

    t_start_f = float(t_start)
    duration_to_bound = float_duration(t_start, t_bound)
    if !(duration_to_bound > 0.)
        return nothing
    end
    if !isfinite(proposed_dt) || proposed_dt >= duration_to_bound
        return (; t_end = t_bound, dt = duration_to_bound)
    end

    # Canonicalizing the absolute proposed time avoids both accumulated floating-point time
    # and repeated rational addition. `solver_time` bounds the denominator complexity of
    # this approximate soft endpoint; exact hard event times never use that conversion.
    proposed_time = t_start_f + proposed_dt
    t_end = solver_time(proposed_time)
    if time_isless(t_bound, t_end)
        return (; t_end = t_bound, dt = duration_to_bound)
    end
    if !time_isless(t_start, t_end)
        return nothing
    end

    # A soft endpoint is an approximate label for a floating-point numerical instant. Its
    # exact rational difference can subtly change an adaptive method's accepted duration,
    # especially during a stiff transient. Difference the numerical labels for soft steps;
    # exact hard-bound intervals use `float_duration` above.
    dt = float(t_end) - t_start_f
    return dt > 0. ? (; t_end, dt) : nothing

end

# `evaluate_remaining_stages` uses value recursion rather than a runtime loop. Each
# recursive call has a different, statically known tuple of rates, so Julia can specialize
# propagation without the type instability caused by repeatedly assigning
# `(stages..., new_stage)` to one local variable in the original DP54 implementation.

@inline function evaluate_remaining_stages(
    method,
    problem,
    t_start_f,
    dt,
    state_at_start,
    stages,
    ::Val{N},
    ::Val{N},
) where {N}
    return stages
end

@inline function evaluate_remaining_stages(
    method,
    problem,
    t_start_f,
    dt,
    state_at_start,
    stages,
    ::Val{I},
    ::Val{N},
) where {I, N}

    gains = map(coefficient -> dt * coefficient, method.a[I])
    stage_state = propagate(state_at_start, gains, stages)
    stage_time = t_start_f + dt * method.c[I]
    stage_rates = evaluate_rates(problem, stage_time, stage_state)

    return evaluate_remaining_stages(
        method,
        problem,
        t_start_f,
        dt,
        state_at_start,
        (stages..., stage_rates),
        Val(I + 1),
        Val(N),
    )

end

function stage_count(
    method::ExplicitRungeKuttaTableau{A, B, C, EB},
) where {A, B, C, EB}
    return Val(fieldcount(C) + 1)
end

"""
Evaluates every stage for one explicit Runge-Kutta attempt over `[t_start, t_end]`. The
beginning state already contains the continuous random values committed for the surrounding
random interval.
"""
function evaluate_stages(
    method::ExplicitRungeKuttaTableau,
    problem::ContinuousProblem,
    t_start::Rational,
    t_end::Rational,
    dt::Float64,
    state_at_start::ModelStateDescription,
)
    t_start_f = float(t_start)
    rates_at_start = evaluate_rates(problem, t_start_f, state_at_start)
    stages = evaluate_remaining_stages(
        method,
        problem,
        t_start_f,
        dt,
        state_at_start,
        (rates_at_start,),
        Val(1),
        stage_count(method),
    )
    return (; state_at_start, rates_at_start, stages, dt)
end

function solution_state(state_at_start, dt, weights, stages)
    gains = map(coefficient -> dt * coefficient, weights)
    return propagate(state_at_start, gains, stages)
end

"""
    step!(integrator, problem, request)

Attempts and returns exactly one accepted numerical step without crossing
`request.t_bound`.
The fixed and adaptive overloads below form the complete simulation-facing solver protocol.
"""
function step!(
    integrator::RungeKuttaIntegrator{M, C},
    problem::ContinuousProblem,
    request::StepRequest,
) where {M, C <: FixedStepController}

    # Figure out how far to step. We've been requested to step all the way to t_bound, but
    # we may need to limit for our specified time step too.
    interval = choose_step_interval(
        request.t_start,
        request.t_bound,
        integrator.controller.dt,
    )
    if isnothing(interval)
        failure = SolverStepSizeUnderflow(
            float(request.t_start),
            integrator.controller.dt,
        )
        return SolverFailure(request.t_start, failure)
    end

    # If it's time to take draws again, do so, and commit to the interval over which we'll
    # hold those draws. Otherwise, pass through the unmodified state, and keep the next CRV
    # draw time.
    if request.t_start == request.t_next_crv_draw
        state_at_start = draw_continuous_random_variables(
            problem, request.t_start, interval.dt, request.state,
        )
        t_next_crv_draw = interval.t_end
    else
        state_at_start = request.state
        t_next_crv_draw = request.t_next_crv_draw
    end

    # Calculate each of the derivatives.
    attempt = evaluate_stages(
        integrator.method,
        problem,
        request.t_start,
        interval.t_end,
        interval.dt,
        state_at_start,
    )

    # Assemble the updated state.
    state_at_end = solution_state(
        attempt.state_at_start,
        attempt.dt,
        integrator.method.b,
        attempt.stages,
    )

    return AcceptedStep(
        interval.t_end,
        attempt.state_at_start,
        attempt.rates_at_start,
        state_at_end,
        integrator.controller.dt,
        t_next_crv_draw,
    )

end

function suggested_dt(controller, dt, error, embedded_order)
    if iszero(error)
        return controller.max_dt
    end
    return min(
        controller.safety_factor * dt * error^(-1 / (embedded_order + 1)),
        controller.max_dt,
    )
end

function step!(
    integrator::RungeKuttaIntegrator{M, C},
    problem::ContinuousProblem,
    request::StepRequest,
) where {M, C <: EmbeddedAdaptiveController}

    controller = integrator.controller

    # Figure out how far to step. We've been requested to step all the way to t_bound, but
    # we may know that we need smaller steps for integration tolerances.
    proposed_dt = min(controller.next_dt, controller.max_dt)
    interval = choose_step_interval(request.t_start, request.t_bound, proposed_dt)
    if isnothing(interval)
        failure = SolverStepSizeUnderflow(float(request.t_start), proposed_dt)
        return SolverFailure(request.t_start, failure)
    end

    # If it's time to take draws again, do so, and commit to the interval over which we'll
    # hold those draws. Otherwise, pass through the unmodified state, and keep the next CRV
    # draw time.
    if request.t_start == request.t_next_crv_draw
        state_at_start = draw_continuous_random_variables(
            problem, request.t_start, interval.dt, request.state,
        )
        t_next_crv_draw = interval.t_end
    else
        state_at_start = request.state
        t_next_crv_draw = request.t_next_crv_draw
    end

    for rejection_count in 0:controller.max_rejections

        # Calculate each of the derivatives.
        attempt = evaluate_stages(
            integrator.method,
            problem,
            request.t_start,
            interval.t_end,
            interval.dt,
            state_at_start,
        )

        # Assemble the updated state and embedded state.
        state_at_end = solution_state(
            attempt.state_at_start,
            attempt.dt,
            integrator.method.b,
            attempt.stages,
        )
        embedded_state = solution_state(
            attempt.state_at_start,
            attempt.dt,
            integrator.method.embedded_b,
            attempt.stages,
        )

        # Figure out how much normalized error we have, and what dt that suggests for next
        # time.
        error = normalized_error(
            problem,
            state_at_end,
            embedded_state,
            controller.absolute_tolerance,
            controller.relative_tolerance,
        )
        next_dt = suggested_dt(
            controller,
            attempt.dt,
            error,
            integrator.method.embedded_order,
        )

        # If that's accept, remember to the next_dt we determined and report an accepted
        # step.
        if error < 1.
            controller.next_dt = next_dt
            return AcceptedStep(
                interval.t_end,
                attempt.state_at_start,
                attempt.rates_at_start,
                state_at_end,
                next_dt,
                t_next_crv_draw,
            )
        end

        # Otherwise, the step failed to meet tolerance. See if it's time to give up.
        if rejection_count == controller.max_rejections
            failure = SolverFailedToConverge(float(request.t_start))
            return SolverFailure(request.t_start, failure)
        end

        # It's not time to give up, so let's modify the proposed time step and try this
        # again.
        proposed_dt = next_dt
        interval = choose_step_interval(request.t_start, request.t_bound, proposed_dt)
        if isnothing(interval)
            failure = SolverStepSizeUnderflow(float(request.t_start), proposed_dt)
            return SolverFailure(request.t_start, failure)
        end

    end

    # The loop above always accepts a step or returns a concrete failure. This final error
    # is an assertion about the implementation rather than a numerical failure mode.
    error("The adaptive step loop completed without an accepted step or failure.")

end

end # Solvers
