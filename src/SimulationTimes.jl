"""
The `SimulationTimes` module defines the boundary between exact event scheduling and
floating-point numerical integration.

User- and model-requested times are official simulation times. They remain exact rationals
so unrelated periodic systems can share one scheduler without clock drift or a common base
step. Solver-selected times between those events are numerical conveniences rather than
physically exact instants. They are converted to rationals only so the exact scheduler has
one concrete time representation; their numerical integration duration remains a float.

All nontrivial arithmetic is centralized here. In particular, converting an interval to a
`Float64` must not first subtract two `Rational{Int64}` values, because the intermediate
cross-products can overflow even when the final duration is small and representable.
"""
module SimulationTimes

export ExactTime, KEEP_T_NEXT, NO_T_NEXT,
    exact_time, solver_time,
    float_duration, earlier_time, time_isless,
    is_regular_step_triggering, next_regular_time

"""
The concrete representation currently used for every official simulation time.

This alias gives the representation one home and makes a future `Int128` experiment local
rather than requiring another search for hard-coded rational types.
"""
const ExactTime = Rational{Int64}

"""
An `UpdatesOutput` instruction to retain the model's previously requested `t_next`.

Negative rational infinity is consumed by the update operation and is never stored as a
model's actual next event time.
"""
const KEEP_T_NEXT = -1//0

"""
The exact scheduler value meaning that a model has no finite upcoming event.

Positive rational infinity participates naturally in ordering and minimum operations. A
model may also request this value explicitly to cancel a previously scheduled event.
"""
const NO_T_NEXT = 1//0

"""
    exact_time(t)

Convert a user- or model-provided time to the official exact representation.

Rational inputs retain their mathematical value. Floating-point inputs use Julia's normal
`rationalize` semantics, which intentionally recover simple values such as `0.1 == 1//10`.
When exact scheduling matters, callers should provide a rational explicitly.
"""
exact_time(t::ExactTime) = t
exact_time(t::Real) = rationalize(Int64, t)

"""
    time_isless(a, b)

Compare two official times using `Int128` intermediates.

Julia's ordinary bounded rational arithmetic may overflow while cross-multiplying unrelated
denominators. Widening here keeps exact scheduler comparisons allocation-free for all
practical `Rational{Int64}` values.

Rational infinities require explicit handling because their zero denominators would make
the finite cross-product comparison meaningless. Their ordering is:

```
-1//0 < every finite time < 1//0
```
"""
function time_isless(a::ExactTime, b::ExactTime)

    a_is_infinite = isinf(a)
    b_is_infinite = isinf(b)

    # When both values are infinite, their numerators distinguish negative from positive
    # infinity. When only one is infinite, it precedes a finite value exactly when it is
    # negative, and a finite value precedes it exactly when it is positive.
    if a_is_infinite
        return b_is_infinite ? numerator(a) < numerator(b) : numerator(a) < 0
    elseif b_is_infinite
        return numerator(b) > 0
    end

    # Both values are finite, so widened cross-multiplication gives an exact comparison
    # without constructing another Rational and without overflowing Int64 intermediates.
    return (
        Int128(numerator(a)) * Int128(denominator(b)) <
        Int128(numerator(b)) * Int128(denominator(a))
    )

end


"""
    earlier_time(a, b)

Return the earlier of two official simulation times using overflow-resistant comparison.
"""
earlier_time(a::ExactTime, b::ExactTime) = time_isless(a, b) ? a : b

"""
    float_duration(t_start, t_end)

Convert the exact interval from `t_start` to `t_end` into a floating-point numerical
duration without first constructing `t_end - t_start`.

The whole and fractional portions are differenced separately using `Int128` intermediates.
This avoids both `Rational{Int64}` overflow and the cancellation that can occur when two
large absolute times are individually rounded to `Float64` before subtraction.
"""
function float_duration(t_start::ExactTime, t_end::ExactTime)

    isfinite(t_start) || throw(ArgumentError("t_start must be finite."))
    isfinite(t_end) || throw(ArgumentError("t_end must be finite."))

    start_whole, start_remainder = divrem(numerator(t_start), denominator(t_start))
    end_whole, end_remainder = divrem(numerator(t_end), denominator(t_end))

    whole_difference = Int128(end_whole) - Int128(start_whole)
    fractional_numerator = (
        Int128(end_remainder) * Int128(denominator(t_start)) -
        Int128(start_remainder) * Int128(denominator(t_end))
    )
    fractional_denominator = (
        Int128(denominator(t_start)) * Int128(denominator(t_end))
    )

    return Float64(whole_difference) +
        Float64(fractional_numerator) / Float64(fractional_denominator)

end


"""
    solver_time(t)

Convert a floating-point solver-selected endpoint to the official time representation.

Unlike an exact hard event, this endpoint has no independent claim to mathematical
exactness. `rationalize` supplies a scheduler label for the floating-point instant without
accumulating time through repeated floating-point or rational addition. Keeping this
conversion in a named function makes the exact-scheduler/numerical-integrator boundary
explicit without imposing an additional approximation policy.
"""
function solver_time(t::Float64)
    isfinite(t) || throw(ArgumentError("A solver-generated endpoint must be finite."))
    return exact_time(t)
end


"""
    wide_time(t)

Widen an official time before exact schedule arithmetic. Calculations involving several
unrelated denominators can then use Julia's ordinary `Rational` operations without
overflowing an `Int64` intermediate.
"""
wide_time(t::ExactTime) = Int128(numerator(t)) // Int128(denominator(t))

"""
    narrow_time(value)

Return a widened calculation to the official time representation.

This check distinguishes an unrepresentable final time from an avoidable overflow in an
intermediate calculation. It is the only custom arithmetic boundary needed by the regular
schedule helpers below.
"""
function narrow_time(value::Rational{Int128})
    numerator(value) >= typemin(Int64) || throw(OverflowError(
        "The exact time numerator is below the Rational{Int64} range.",
    ))
    numerator(value) <= typemax(Int64) || throw(OverflowError(
        "The exact time numerator is above the Rational{Int64} range.",
    ))
    denominator(value) <= typemax(Int64) || throw(OverflowError(
        "The exact time denominator is above the Rational{Int64} range.",
    ))
    return Int64(numerator(value)) // Int64(denominator(value))
end


"""
    next_regular_time(t, period, offset = 0//1)

Return the first time in `offset + n * period`, for nonnegative integer `n`, that is
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

Return whether exact time `t` belongs to the periodic sequence `offset + n * period`.

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

end # SimulationTimes
