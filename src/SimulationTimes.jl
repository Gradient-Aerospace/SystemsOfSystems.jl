"""
The `SimulationTimes` module defines the boundary between exact event scheduling and
floating-point numerical integration.

User- and model-requested times are official simulation times. They remain exact rationals
so unrelated periodic systems can share one scheduler without clock drift or a common base
step. Solver-selected times between those events are numerical conveniences rather than
physically exact instants. They are therefore converted to rationals with deliberately
bounded denominator complexity.

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
The largest denominator permitted for a solver-generated soft endpoint.

Hard event times supplied as rationals are never approximated or constrained by this value.
The effective limit may be smaller at very large absolute times so the resulting numerator
continues to fit in `Int64`.
"""
const MAX_SOLVER_TIME_DENOMINATOR = Int64(1_000_000_000_000_000_000)

"""
    exact_time(t)

Convert a user- or model-provided time to the official exact representation.

Rational inputs retain their mathematical value. Floating-point inputs use Julia's normal
`rationalize` semantics, which intentionally recover simple values such as `0.1 == 1//10`.
When exact scheduling matters, callers should provide a rational explicitly.
"""
exact_time(t::ExactTime) = t
exact_time(t::Real) = rationalize(Int64, t)

# Rational infinities require explicit ordering because the widened finite cross-product
# formula would multiply both values by zero denominators.
function infinity_class(t::ExactTime)
    if !iszero(denominator(t))
        return 0
    end
    return sign(numerator(t))
end

"""
    time_isless(a, b)

Compare two official times using `Int128` intermediates.

Julia's ordinary bounded rational arithmetic may overflow while cross-multiplying unrelated
denominators. Widening here keeps exact scheduler comparisons allocation-free for all
practical `Rational{Int64}` values.
"""
function time_isless(a::ExactTime, b::ExactTime)

    a_class = infinity_class(a)
    b_class = infinity_class(b)
    if !iszero(a_class) || !iszero(b_class)
        return a_class < b_class
    end

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

    iszero(denominator(t_start)) && throw(ArgumentError("t_start must be finite."))
    iszero(denominator(t_end)) && throw(ArgumentError("t_end must be finite."))

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


# Choose a denominator limit that also leaves enough numerator range for the absolute time.
function solver_denominator_limit(t::Float64)

    isfinite(t) || throw(ArgumentError("A solver-generated endpoint must be finite."))
    abs(t) <= typemax(Int64) || throw(OverflowError(
        "The solver-generated endpoint $t cannot be represented as Rational{Int64}.",
    ))

    integer_magnitude = max(Int64(1), ceil(Int64, abs(t)))
    numerator_limit = max(Int64(1), typemax(Int64) ÷ integer_magnitude)
    return min(MAX_SOLVER_TIME_DENOMINATOR, numerator_limit)

end


"""
    solver_time(t)

Convert a floating-point solver-selected endpoint to a bounded-complexity official time.

Unlike an exact hard event, this endpoint has no independent claim to mathematical
exactness. Its denominator is limited so repeated adaptive steps cannot introduce enormous
cross-products into the exact event scheduler. The conversion begins at a tolerance suited
to the requested denominator and relaxes it only when necessary to satisfy the hard limit.
"""
function solver_time(t::Float64)

    denominator_limit = solver_denominator_limit(t)
    tolerance = max(eps(t), 0.5 / Float64(denominator_limit)^2)
    result = rationalize(Int64, t; tol = tolerance)

    while denominator(result) > denominator_limit
        tolerance *= 2
        result = rationalize(Int64, t; tol = tolerance)
    end

    return result

end


# The following exact arithmetic helpers use Int128 only for intermediate products and then
# return the established ExactTime representation. They fail only if the reduced result
# itself cannot be represented, rather than because an avoidable Int64 intermediate wrapped.

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


function subtract_time(a::ExactTime, b::ExactTime)
    value = (
        Int128(numerator(a)) // Int128(denominator(a)) -
        Int128(numerator(b)) // Int128(denominator(b))
    )
    return narrow_time(value)
end


function add_time(a::ExactTime, b::ExactTime)
    value = (
        Int128(numerator(a)) // Int128(denominator(a)) +
        Int128(numerator(b)) // Int128(denominator(b))
    )
    return narrow_time(value)
end


function scale_time(value::ExactTime, multiplier::Integer)

    common_factor = gcd(abs(Int128(multiplier)), Int128(denominator(value)))
    reduced_multiplier = Int128(multiplier) ÷ common_factor
    reduced_denominator = Int128(denominator(value)) ÷ common_factor
    result = (
        reduced_multiplier * Int128(numerator(value)) //
        reduced_denominator
    )
    return narrow_time(result)

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
    period_exact > 0 || throw(ArgumentError("period must be positive."))

    if time_isless(t_exact, offset_exact)
        return offset_exact
    end

    elapsed = subtract_time(t_exact, offset_exact)
    elapsed_wide = Int128(numerator(elapsed)) // Int128(denominator(elapsed))
    period_wide = Int128(numerator(period_exact)) // Int128(denominator(period_exact))
    sample_index = fld(elapsed_wide, period_wide) + 1

    return add_time(offset_exact, scale_time(period_exact, sample_index))

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
    iszero(period_exact) && return true
    period_exact > 0 || throw(ArgumentError("period cannot be negative."))
    time_isless(t_exact, offset_exact) && return false

    elapsed = subtract_time(t_exact, offset_exact)
    elapsed_wide = Int128(numerator(elapsed)) // Int128(denominator(elapsed))
    period_wide = Int128(numerator(period_exact)) // Int128(denominator(period_exact))
    return isinteger(elapsed_wide / period_wide)

end

end # SimulationTimes
