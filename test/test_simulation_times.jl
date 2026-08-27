module TestSimulationTimes

using Test
using SystemsOfSystems

const SimulationTimes = SystemsOfSystems.SimulationTimes

@testset "overflow-safe floating durations" begin

    # These two perfectly valid Rational{Int64} values have unrelated large denominators.
    # Ordinary rational subtraction overflows while forming cross-products, even though the
    # small positive difference is readily representable as a Float64.
    t_start = 146_990_887_513_614 // 146_990_887_513_615
    t_end = 1_921_204_408_937 // 1_921_204_408_936
    @test_throws OverflowError t_end - t_start
    @test SimulationTimes.float_duration(t_start, t_end) > 0.

    # Converting large absolute times to Float64 before subtraction can erase a small
    # interval. The centralized conversion differences exact whole and fractional portions
    # first, so the numerical duration retains the requested tenth of a second.
    large_start = 9_000_000_000_000_000//1
    large_end = large_start + 1//10
    @test float(large_end) - float(large_start) == 0.
    @test SimulationTimes.float_duration(large_start, large_end) == 0.1

end

@testset "solver endpoint conversion" begin

    # Solver-selected endpoints originate as floating-point approximations. Rationalizing
    # the absolute endpoint gives the exact scheduler one time representation without
    # claiming that the label is an independently exact event.
    proposed_time = 0.000_012_345_678_901_234_5
    t = SimulationTimes.solver_time(proposed_time)
    @test t == rationalize(Int64, proposed_time)
    @test_throws ArgumentError SimulationTimes.solver_time(Inf)

    # Simple decimal clock times remain pleasantly simple even at a large epoch.
    t = SimulationTimes.solver_time(1_000_000_000.1)
    @test t == 10_000_000_001//10

    # Soft intervals use their floating-point endpoint labels for numerical integration;
    # their rational values are scheduler labels rather than claims of exact physical time.
    soft_interval = SystemsOfSystems.Solvers.choose_step_interval(0//1, 1//1, 1.6e-13)
    @test soft_interval.t_end > 0//1
    @test soft_interval.dt == float(soft_interval.t_end)

    # Hard event intervals instead derive dt from the exact rational endpoints. This still
    # works when the absolute Float64 endpoint labels are identical.
    hard_start = 9_000_000_000_000_000//1
    hard_bound = hard_start + 1//10
    hard_interval = SystemsOfSystems.Solvers.choose_step_interval(
        hard_start,
        hard_bound,
        1.,
    )
    @test hard_interval.t_end == hard_bound
    @test hard_interval.dt == 0.1

    # A non-advancing hard bound must be rejected before attempt preparation can draw a
    # continuous random variable over a zero-duration interval.
    @test isnothing(SystemsOfSystems.Solvers.choose_step_interval(1//1, 1//1, 1.))

end

@testset "requested simulation times" begin

    # Input validation happens before initialization so invalid requests cannot open model
    # resources or reach the continuous solver. Equal adjacent times are invalid as well as
    # decreasing times because every accepted interval must have positive duration.
    init_was_called = Ref(false)
    init = (args...) -> begin
        init_was_called[] = true
        return ModelDescription()
    end

    @test_throws ArgumentError simulate(nothing; t = (), init_fcn = init)
    @test_throws ArgumentError simulate(nothing; t = (0,), init_fcn = init)
    @test_throws ArgumentError simulate(nothing; t = (0, 0), init_fcn = init)
    @test_throws ArgumentError simulate(nothing; t = (0, 2, 1), init_fcn = init)
    @test_throws ArgumentError simulate(nothing; t = (0, NO_T_NEXT), init_fcn = init)
    @test !init_was_called[]

end

@testset "overflow-resistant time ordering" begin

    # The scheduler sentinels have the ordinary extended-real ordering. These cases are
    # handled explicitly because cross-multiplication cannot distinguish values whose
    # Rational denominator is zero.
    @test SimulationTimes.time_isless(KEEP_T_NEXT, -1//1)
    @test SimulationTimes.time_isless(-1//1, NO_T_NEXT)
    @test SimulationTimes.time_isless(KEEP_T_NEXT, NO_T_NEXT)
    @test !SimulationTimes.time_isless(NO_T_NEXT, KEEP_T_NEXT)
    @test !SimulationTimes.time_isless(NO_T_NEXT, NO_T_NEXT)

    # Unrelated large finite denominators use widened cross-products rather than the
    # potentially overflowing Rational{Int64} comparison implementation.
    earlier = 146_990_887_513_614 // 146_990_887_513_615
    later = 1_921_204_408_937 // 1_921_204_408_936
    @test SimulationTimes.time_isless(earlier, later)

end

@testset "stateless regular schedules" begin

    # `next_regular_time` is strictly later than its input because initialization
    # establishes the model at t_start without performing a discrete update there.
    @test next_regular_time(0//1, 1//10) == 1//10
    @test next_regular_time(1//10, 1//10) == 1//5

    # An offset is the first occurrence. Before it, the next time is exactly the offset;
    # afterwards, the closed-form calculation finds the first later occurrence without a
    # mutable sample counter.
    @test next_regular_time(0//1, 1//10, 1//30) == 1//30
    @test next_regular_time(1//30, 1//10, 1//30) == 2//15
    @test next_regular_time(1//10, 1//3, 1//7) == 1//7

    # Rational infinities are scheduler instructions, not meaningful components of a
    # periodic sequence.
    @test_throws ArgumentError next_regular_time(NO_T_NEXT, 1//10)
    @test_throws ArgumentError next_regular_time(0//1, NO_T_NEXT)
    @test_throws ArgumentError is_regular_step_triggering(0//1, 1//10, NO_T_NEXT)

end

@testset "time request sentinels" begin

    @test ModelDescription().t_next == NO_T_NEXT
    @test UpdatesOutput().t_next == KEEP_T_NEXT
    @test SystemsOfSystems.update_model_t_next(1//2, KEEP_T_NEXT) == 1//2
    @test SystemsOfSystems.update_model_t_next(1//2, NO_T_NEXT) == NO_T_NEXT
    @test SystemsOfSystems.update_model_t_next(1//2, 3//4) == 3//4

end

@testset "nothing is a no-op discrete result" begin

    # Returning `nothing` must preserve the exact ModelStateDescription object. Besides
    # defining the result semantically, object identity guards the intended allocation-free
    # fast path. Stop traversal must likewise interpret it as no request.
    model_state = SystemsOfSystems.ModelStateDescription{Nothing}(;
        discrete_states = (; count = 1,),
        t_next = 1//2,
    )

    @test SystemsOfSystems.update(model_state, nothing) === model_state
    @test isnothing(SystemsOfSystems.find_model_requested_stop(nothing))

    # Empty UpdatesOutput values remain valid for compatibility, although only `nothing`
    # promises to return the original object without traversing its contents.
    updated_model_state = SystemsOfSystems.update(model_state, UpdatesOutput())
    @test updated_model_state.discrete_states == model_state.discrete_states
    @test updated_model_state.t_next == model_state.t_next
    @test isnothing(SystemsOfSystems.find_model_requested_stop(UpdatesOutput()))

end

@testset "independent exact model clocks at a large epoch" begin

    # Two nested models request unrelated periods and offsets. Each model owns only its
    # ordinary discrete state; its next requested time remains scheduler metadata managed by
    # SystemsOfSystems. There is no common base step and no update at t_start.
    t_start = 1_000_000_000//1
    t_end = t_start + 1//1
    period_a = 1//10
    period_b = 1//3
    offset_a = t_start + 1//30
    offset_b = t_start + 1//21
    triggers_a = Rational{Int64}[]
    triggers_b = Rational{Int64}[]

    history = simulate(
        nothing;
        t = (t_start, t_end),
        init_fcn = (args...) -> ModelDescription(;
            models = (;
                a = ModelDescription(;
                    discrete_states = (; n = 0,),
                    t_next = next_regular_time(t_start, period_a, offset_a),
                ),
                b = ModelDescription(;
                    discrete_states = (; n = 0,),
                    t_next = next_regular_time(t_start, period_b, offset_b),
                ),
            ),
        ),
        updates_fcn = (t, model) -> begin

            update_a = if is_regular_step_triggering(t, period_a, offset_a)
                push!(triggers_a, t)
                UpdatesOutput(;
                    updates = (; n = model.a.n + 1,),
                    t_next = next_regular_time(t, period_a, offset_a),
                )
            else
                nothing
            end
            update_b = if is_regular_step_triggering(t, period_b, offset_b)
                push!(triggers_b, t)
                UpdatesOutput(;
                    updates = (; n = model.b.n + 1,),
                    t_next = next_regular_time(t, period_b, offset_b),
                )
            else
                nothing
            end

            return UpdatesOutput(; models = (; a = update_a, b = update_b,),)

        end,
    )

    expected_a = collect(offset_a:period_a:t_end)
    expected_b = collect(offset_b:period_b:t_end)
    @test history.t_stop == t_end
    @test triggers_a == expected_a
    @test triggers_b == expected_b
    @test history.model.a.n == length(expected_a)
    @test history.model.b.n == length(expected_b)
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

end # TestSimulationTimes
