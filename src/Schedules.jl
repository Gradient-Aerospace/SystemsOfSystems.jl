"""
The `Schedules` module defines declarative event-time patterns used by models and the
simulation scheduler.

Schedules answer two exact-time questions: whether a particular accepted sample is one of
their occurrences, and when their next occurrence will be. They depend on
`SimulationTimes` for overflow-resistant rational arithmetic but remain independent of
model descriptions, `UpdatesOutput`, logging, and numerical solvers. That boundary lets
users add schedule types without coupling them to the rest of the simulation engine.
"""
module Schedules

import Dimensions

using ..SimulationTimes: ExactTime, NO_T_NEXT,
    exact_time, earlier_time, narrow_time, time_isless, wide_time

export AbstractSchedule, RegularSchedule, OffsetRegularSchedule,
    is_triggering, next_trigger_time,
    is_regular_step_triggering, next_regular_time

"""
    AbstractSchedule

The common supertype for declarative model schedules.

A schedule is immutable initialization metadata describing times at which the simulation
must produce accepted samples. Custom schedules implement:

* `is_triggering(schedule, t)`, which reports whether official time `t` is an occurrence.
* `next_trigger_time(schedule, t)`, which returns the first occurrence strictly later than
  `t`, or `NO_T_NEXT` when the schedule has no later occurrence.

Schedules are deduplicated by value during initialization. Custom schedule types should
therefore provide consistent value-based `isequal` and `hash` methods when Julia's defaults
do not already express their desired identity.
"""
abstract type AbstractSchedule end

"""
    is_triggering(schedule::AbstractSchedule, t)

Returns whether `schedule` has an occurrence at official simulation time `t`.

Implementations should use exact-time comparisons. The simulation calls every model's
update function after every accepted step, so models use this predicate to distinguish
their scheduled samples from unrelated solver, user, or model event times.
"""
function is_triggering end

"""
    next_trigger_time(schedule::AbstractSchedule, t)

Returns the first occurrence of `schedule` strictly later than official simulation time
`t`.

The strict inequality is part of the interface: initialization establishes the model at
`t_start` without performing a discrete update there. A finite schedule may return
`NO_T_NEXT` when it has no remaining occurrences.
"""
function next_trigger_time end

"""
    RegularSchedule(; period)
    RegularSchedule(period)

A schedule occurring at `n * period` for every nonnegative integer `n`.

`period` is stored as an exact `Rational{Int64}` and must be finite and strictly positive.
The representation stores no offset, making this common schedule both compact and direct
to evaluate.
"""
struct RegularSchedule <: AbstractSchedule

    period::ExactTime

    function RegularSchedule(period)
        period = exact_time(period)
        isfinite(period) || throw(ArgumentError("period must be finite."))
        period > 0 || throw(ArgumentError("period must be positive."))
        return new(period)
    end

end
RegularSchedule(; period) = RegularSchedule(period)

Dimensions.dimstyle(::Type{RegularSchedule}) = Dimensions.StructDimensionStyle()

"""
    OffsetRegularSchedule(; period, offset)
    OffsetRegularSchedule(period, offset)

A schedule occurring at `offset + n * period` for every nonnegative integer `n`.

The finite `offset` is the first occurrence, not merely a phase extended backward without
limit. `period` is exact, finite, and strictly positive.
"""
struct OffsetRegularSchedule <: AbstractSchedule

    period::ExactTime
    offset::ExactTime

    function OffsetRegularSchedule(period, offset)
        period = exact_time(period)
        offset = exact_time(offset)
        isfinite(period) || throw(ArgumentError("period must be finite."))
        isfinite(offset) || throw(ArgumentError("offset must be finite."))
        period > 0 || throw(ArgumentError("period must be positive."))
        return new(period, offset)
    end

end
OffsetRegularSchedule(; period, offset) = OffsetRegularSchedule(period, offset)

Dimensions.dimstyle(::Type{OffsetRegularSchedule}) = Dimensions.StructDimensionStyle()

"""
    next_regular_time(t, period, offset = 0//1)

Returns the first time in `offset + n * period`, for nonnegative integer `n`, that is
strictly later than `t`.

This function is intentionally not inclusive: `init_fcn` establishes the model at the
simulation start time, and SystemsOfSystems does not perform a discrete update at `t_start`.
The calculation is closed-form and therefore needs no mutable sample index or accumulated
floating-point clock.
"""
function next_regular_time(t, period, offset = 0//1)

    t_exact = exact_time(t)
    period_exact = exact_time(period)
    offset_exact = exact_time(offset)
    isfinite(t_exact) || throw(ArgumentError("t must be finite."))
    isfinite(period_exact) || throw(ArgumentError("period must be finite."))
    isfinite(offset_exact) || throw(ArgumentError("offset must be finite."))
    period_exact > 0 || throw(ArgumentError("period must be positive."))

    if time_isless(t_exact, offset_exact)
        return offset_exact
    end

    # Perform the whole calculation with ordinary Rational{Int128} arithmetic, then check
    # only the final result when returning to the official Rational{Int64} representation.
    t_wide = wide_time(t_exact)
    period_wide = wide_time(period_exact)
    offset_wide = wide_time(offset_exact)
    sample_index = fld(t_wide - offset_wide, period_wide) + 1

    return narrow_time(offset_wide + sample_index * period_wide)

end

"""
    is_regular_step_triggering(t, period, offset = 0//1)

Returns whether exact time `t` belongs to the periodic sequence `offset + n * period`.

A zero period retains the existing convention of triggering at every accepted sample. For a
positive period, widened exact arithmetic avoids floating-point tolerances and bounded
rational intermediate overflow.
"""
function is_regular_step_triggering(t, period, offset = 0//1)

    t_exact = exact_time(t)
    period_exact = exact_time(period)
    offset_exact = exact_time(offset)
    isfinite(t_exact) || throw(ArgumentError("t must be finite."))
    isfinite(period_exact) || throw(ArgumentError("period must be finite."))
    isfinite(offset_exact) || throw(ArgumentError("offset must be finite."))
    iszero(period_exact) && return true
    period_exact > 0 || throw(ArgumentError("period cannot be negative."))
    time_isless(t_exact, offset_exact) && return false

    elapsed_wide = wide_time(t_exact) - wide_time(offset_exact)
    period_wide = wide_time(period_exact)
    return isinteger(elapsed_wide / period_wide)

end

# The public schedule methods share the compatibility helpers' widened exact arithmetic so
# the package has one mathematical definition of a regular clock.
is_triggering(schedule::RegularSchedule, t) =
    is_regular_step_triggering(t, schedule.period)

is_triggering(schedule::OffsetRegularSchedule, t) =
    is_regular_step_triggering(t, schedule.period, schedule.offset)

next_trigger_time(schedule::RegularSchedule, t) =
    next_regular_time(t, schedule.period)

next_trigger_time(schedule::OffsetRegularSchedule, t) =
    next_regular_time(t, schedule.period, schedule.offset)

"""
    find_soonest_time(schedules, t_last)

Returns the first declared schedule occurrence strictly later than `t_last`.

Tuple mapping preserves specialization for every concrete schedule type and produces a
tuple of exact next times. This is intentionally simple for the initial architecture; very
large tuples may eventually justify a cached scheduler or another runtime index.
"""
function find_soonest_time(schedules::Tuple, t_last)

    schedule_times = map(schedules) do schedule

        schedule_time = exact_time(next_trigger_time(schedule, t_last))
        time_isless(t_last, schedule_time) ||
            throw_invalid_next_trigger_time(schedule, t_last, schedule_time)
        return schedule_time

    end

    return reduce(earlier_time, schedule_times; init = NO_T_NEXT)

end

# Keep detailed diagnostic construction off the successful hot path. Custom schedules are
# validated on every query because returning the current or a past time would otherwise
# prevent the simulation loop from making progress.
Base.@noinline function throw_invalid_next_trigger_time(schedule, t_last, schedule_time)
    throw(ArgumentError(
        "$(typeof(schedule)) returned $schedule_time from next_trigger_time at " *
        "t = $t_last; schedule times must be strictly later than t.",
    ))
end

end # Schedules
