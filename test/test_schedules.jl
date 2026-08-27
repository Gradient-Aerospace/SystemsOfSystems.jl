module TestSchedules

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

"""
A small user-defined finite schedule used to verify that the public `AbstractSchedule`
interface works without teaching SystemsOfSystems about every concrete schedule type.
"""
struct ExplicitTestSchedule{T} <: AbstractSchedule
    times::T
end

Base.isequal(a::ExplicitTestSchedule, b::ExplicitTestSchedule) = isequal(a.times, b.times)
Base.hash(schedule::ExplicitTestSchedule, hash_value::UInt) =
    hash(schedule.times, hash_value)

function SystemsOfSystems.is_triggering(schedule::ExplicitTestSchedule, t)
    return t in schedule.times
end

function SystemsOfSystems.next_trigger_time(schedule::ExplicitTestSchedule, t)
    for trigger_time in schedule.times
        if t < trigger_time
            return trigger_time
        end
    end
    return NO_T_NEXT
end

"""
A typed model form proving that named schedules are supplied through the same keyword-based
construction boundary as constants, states, nested models, and resources.
"""
Base.@kwdef struct ScheduledTestModel{S}
    clock::S
    count::Int
end

@testset "regular schedule interface" begin

    regular = RegularSchedule(; period = 1//10)
    offset = OffsetRegularSchedule(; period = 1//10, offset = 1//20)

    # Both built-in forms canonicalize their inputs to exact official times. The common
    # no-offset form also has the intentionally smaller representation discussed in the
    # schedule design.
    @test regular.period == 1//10
    @test offset.period == 1//10
    @test offset.offset == 1//20
    @test sizeof(regular) < sizeof(offset)
    @test RegularSchedule(; period = 0.1) == regular

    # `next_trigger_time` is strictly later than its argument. Although zero belongs to the
    # regular schedule, simulation initialization does not perform an update at t_start.
    @test is_triggering(regular, 0//1)
    @test is_triggering(regular, 3//10)
    @test !is_triggering(regular, 1//20)
    @test next_trigger_time(regular, 0//1) == 1//10
    @test next_trigger_time(regular, 1//10) == 1//5

    # An offset is the first occurrence rather than a phase extended backward forever.
    @test !is_triggering(offset, 0//1)
    @test is_triggering(offset, 1//20)
    @test is_triggering(offset, 3//20)
    @test next_trigger_time(offset, 0//1) == 1//20
    @test next_trigger_time(offset, 1//20) == 3//20

    # A schedule must always be able to identify a later finite occurrence, so zero,
    # negative, and infinite periods are invalid declarations.
    @test_throws ArgumentError RegularSchedule(; period = 0)
    @test_throws ArgumentError RegularSchedule(; period = -1//10)
    @test_throws ArgumentError RegularSchedule(; period = NO_T_NEXT)
    @test_throws ArgumentError OffsetRegularSchedule(1//10, NO_T_NEXT)

end

@testset "on-triggering update helper" begin

    schedule = RegularSchedule(; period = 1//10)
    closure_calls = Ref(0)

    # An unrelated accepted sample produces `nothing` and, importantly, does not evaluate
    # the model's scheduled-update body.
    inactive_update = on_triggering(schedule, 1//20) do
        closure_calls[] += 1
        return UpdatesOutput(; updates = (; count = 1,))
    end
    @test iszero(closure_calls[])
    @test isnothing(inactive_update)

    # At a schedule occurrence, the helper evaluates the body exactly once and preserves
    # its complete UpdatesOutput rather than interpreting or rebuilding it.
    active_update = on_triggering(schedule, 1//10) do
        closure_calls[] += 1
        return UpdatesOutput(;
            updates = (; count = 1,),
            outputs = (; sampled = true,),
            stop = true,
        )
    end
    @test closure_calls[] == 1
    @test active_update.updates == (; count = 1,)
    @test active_update.outputs == (; sampled = true,)
    @test active_update.stop

end

@testset "schedules are named model members" begin

    clock = RegularSchedule(; period = 1//5)
    description = ModelDescription(;
        type = ScheduledTestModel,
        discrete_states = (; count = 3,),
        schedules = (; clock,),
    )
    model = initialize(description)

    @test model isa ScheduledTestModel
    @test model.clock == clock
    @test model.count == 3

    # A schedule name cannot silently shadow another value that would be exposed on the
    # same model form, and every value in the schedules block must honor the public type
    # contract.
    conflicting_description = ModelDescription(;
        constants = (; clock = "not a schedule",),
        schedules = (; clock,),
    )
    invalid_description = ModelDescription(;
        schedules = (; clock = 1//5,),
    )
    @test_throws ArgumentError initialize(conflicting_description)
    @test_throws ArgumentError initialize(invalid_description)

end

@testset "hierarchical schedules are collected uniquely" begin

    regular = RegularSchedule(; period = 1//5)
    offset = OffsetRegularSchedule(; period = 1//3, offset = 1//10)
    explicit = ExplicitTestSchedule((1//7, 5//7))
    description = ModelDescription(;
        schedules = (; root_clock = regular,),
        models = (;
            repeated = ModelDescription(; schedules = (; clock = regular,)),
            offset = ModelDescription(; schedules = (; clock = offset,)),
            explicit = ModelDescription(; schedules = (; clock = explicit,)),
        ),
    )

    context = SystemsOfSystems.initialization_context()
    initialized = SystemsOfSystems.create_initialization_artifacts(description, context)

    # Repeated value-equal clocks become one global scheduler entry, while every model
    # retains its own named declaration for local triggering logic.
    @test initialized.schedules == (regular, offset, explicit)
    @test initialized.msd.schedules.root_clock == regular
    @test initialized.msd.models.repeated.schedules.clock == regular
    @test initialized.msd.models.offset.schedules.clock == offset
    @test initialized.msd.models.explicit.schedules.clock == explicit

end

@testset "declared schedules create exact hard sample times" begin

    regular = RegularSchedule(; period = 1//5)
    offset = OffsetRegularSchedule(; period = 1//3, offset = 1//10)
    explicit = ExplicitTestSchedule((1//7, 5//7))
    regular_triggers = Rational{Int64}[]
    repeated_triggers = Rational{Int64}[]
    offset_triggers = Rational{Int64}[]
    explicit_triggers = Rational{Int64}[]

    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            models = (;
                regular = ModelDescription(;
                    discrete_states = (; count = 0,),
                    schedules = (; clock = regular,),
                ),
                repeated = ModelDescription(;
                    discrete_states = (; count = 0,),
                    schedules = (; clock = regular,),
                ),
                offset = ModelDescription(;
                    discrete_states = (; count = 0,),
                    schedules = (; clock = offset,),
                ),
                explicit = ModelDescription(;
                    discrete_states = (; count = 0,),
                    schedules = (; clock = explicit,),
                ),
            ),
        ),
        updates_fcn = (t, model) -> begin

            regular_update = on_triggering(model.regular.clock, t) do
                push!(regular_triggers, t)
                UpdatesOutput(; updates = (; count = model.regular.count + 1,),)
            end
            repeated_update = on_triggering(model.repeated.clock, t) do
                push!(repeated_triggers, t)
                UpdatesOutput(; updates = (; count = model.repeated.count + 1,),)
            end
            offset_update = on_triggering(model.offset.clock, t) do
                push!(offset_triggers, t)
                UpdatesOutput(; updates = (; count = model.offset.count + 1,),)
            end
            explicit_update = on_triggering(model.explicit.clock, t) do
                push!(explicit_triggers, t)
                UpdatesOutput(; updates = (; count = model.explicit.count + 1,),)
            end

            return UpdatesOutput(; models = (;
                regular = regular_update,
                repeated = repeated_update,
                offset = offset_update,
                explicit = explicit_update,
            ))

        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    expected_regular = collect(1//5:1//5:1//1)
    expected_offset = [1//10, 13//30, 23//30]
    expected_explicit = [1//7, 5//7]
    @test history.t_stop == 1
    @test regular_triggers == expected_regular
    @test repeated_triggers == expected_regular
    @test offset_triggers == expected_offset
    @test explicit_triggers == expected_explicit
    @test history.model.regular.count == length(expected_regular)
    @test history.model.repeated.count == length(expected_regular)
    @test history.model.offset.count == length(expected_offset)
    @test history.model.explicit.count == length(expected_explicit)
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

end # TestSchedules
