"""
This module holds things related to TimeSeries. We can't call it TimeSeries because that's
the name of a type it exports.
"""
module TimeSeriesStuff

export Dimension, TimeSeries, AbstractTimeSeriesInterpolator,
    SampleAndHold, LinearInterpolation,
    plot_ts, plot_ts!

using Dimensions: numdims_for_type

"""
    Dimension(; label = "", units = "")

A container for the label and units of one time-series dimension.
"""
struct Dimension
    label::String
    units::String
end
Dimension(; label = "", units = "") = Dimension(label, units)
Base.convert(::Type{Dimension}, pair::Pair) = Dimension(pair.first, pair.second)

"""
    AbstractTimeSeriesInterpolator

Built-in interpolators subtype this abstract type, but `TimeSeries` only requires the
`interpolator` field to be callable as `interpolator(ts, t)`.

That call is made for every requested time, including exact sample times and endpoints.
Passing the full `TimeSeries` and the requested time keeps the interface open for future
interpolators that carry derivative tables, dense-output coefficients, extrapolation
rules, or other per-sample state.
"""
abstract type AbstractTimeSeriesInterpolator end

"""
    SampleAndHold()

An interpolation policy that returns the sample at an exact sample time, or the last stored
sample before the requested time. This is the default for discrete `TimeSeries` values.
"""
struct SampleAndHold <: AbstractTimeSeriesInterpolator end

"""
    LinearInterpolation()

An interpolation policy that linearly interpolates between the samples on either side of
the requested time. This is the default for continuous `TimeSeries` values.
"""
struct LinearInterpolation <: AbstractTimeSeriesInterpolator end

# Users often think in terms of "the LinearInterpolation interpolator", so accepting either
# `LinearInterpolation` or `LinearInterpolation()` keeps the constructor ergonomic. Custom
# callable objects with fields should be passed as instances.
normalize_interpolator(interpolator::Type) = interpolator()
normalize_interpolator(interpolator) = interpolator

# Keep discrete behavior dominant. If the user wants a discrete series to interpolate
# differently, they can still pass an explicit interpolator.
default_interpolator(_el_type, discrete::Bool) = discrete ? SampleAndHold() : LinearInterpolation()

# Mirror the logger's "ignore missing samples" policy for interpolation defaults. If a
# caller stores `Union{Missing, T}` data but the actual samples are present, choose the
# interpolator for `T`.
interpolation_value_type(::Type{Union{Missing, T}}) where {T} = T
interpolation_value_type(::Type{T}) where {T} = T

"""
    TimeSeries

A container for a series of points over time.

Fields:

* `title`: What this time series represents, used as a title in plots
* `time`: An array of times for each element of data stored
* `data`: The array of data, with the same length as `time`
* `time_dimension`: A `Dimension` for time, used as the x-axis label in plots
* `dimensions`: A vector of `Dimension`, one for each dimension of the `data`
* `path`: The model path leading up to this time series (e.g., "/aircraft/imu")
* `discrete`: True if this time series is discrete and false if continuous
* `interpolator`: Callable policy used to evaluate the time series at a requested time
* `groups`: Controls how dimensions are grouped into axes in plots (see below)

The dimension groups should be structured like so:

```
TimeSeries(;
    ...
    dimensions = ["X Pos." => "m", "Y Pos." => "m", "X Vel." => "m/s", "Y Vel." => "m/s"]
    groups = [
        "My Axis 1 Label" => ["X Pos.", "Y Pos."],
        "My Axis 2 Label" => ["X Vel.", "Y Vel."],
    ]
)
```

That is, `groups` is a Vector of Pairs, where each Pair is the name of an axis and an array
of dimension labels that map to that axis. When plotted with `plot_ts`, this example will
result in a figure with two axes, each of which as two lines.
"""
struct TimeSeries{TVT, DVT, IT}
    title::String
    time::TVT
    data::DVT
    time_dimension::Dimension
    dimensions::Vector{Dimension}
    path::String
    discrete::Bool
    interpolator::IT
    groups::Vector{Pair{String, Vector{String}}}
end

# When the user provides no dimensions, we'll rely on the Dimensions interface to provide
# them. We'll label each with the dimension number, and the units will be empty.
function make_default_dimensions(el_type)
    return Dimension[
        Dimension(string(k), "")
        for k in 1:numdims_for_type(el_type)
    ]
end

# When the groups aren't specified, assume we'll want one axis per dimension. We'll label
# each dimension simply with its dimension number, and the units will be empty.
function make_default_groups(dimensions)
    return [
        dim.label => [dim.label,]
        for dim in(dimensions)
    ]
end

function TimeSeries(;
    title::AbstractString,
    time,
    data,
    time_dimension,
    dimensions = missing,
    path::String,
    discrete::Bool = false,
    interpolator = missing,
    groups = missing,
)

    if !isa(time_dimension, Dimension)
        time_dimension = convert(Dimension, time_dimension)
    end

    if ismissing(dimensions)
        dimensions = make_default_dimensions(eltype(data))
    else
        dimensions = Dimension[dimensions...,]
    end

    if ismissing(groups)
        groups = make_default_groups(dimensions)
    end

    if ismissing(interpolator)
        el_type = interpolation_value_type(eltype(data))
        interpolator = default_interpolator(el_type, discrete)
    else
        interpolator = normalize_interpolator(interpolator)
    end

    # Make sure all dimension labels are unique.
    labels = [dim.label for dim in dimensions]
    for label in labels
        if count(==(label), labels) > 1
            error("Dimension labels must be unique, but the $label label was used multiple times in the $title time series (path = $path).")
        end
    end

    # Make sure the groups are valid.
    for (group_label, group_dimension_labels) in groups

        # Make sure there are no degenerate groups.
        @assert !isempty(group_dimension_labels) "A dimension group was empty for the $title time series (path = $path), but this is not allowed."

        # Make sure all of the dimensions in the dimension groups reference valid labels.
        for dim_label in group_dimension_labels
            if count(==(dim_label), labels) != 1
                error("The $group_label dimension group of the $title time series (path = $path) referenced a dimension labeled $dim_label, but there is no dimension with that label. Valid labels: $labels.")
            end
        end

    end

    return TimeSeries{typeof(time), typeof(data), typeof(interpolator)}(
        title, time, data, time_dimension, dimensions, path, discrete, interpolator, groups,
    )

end

"""
    getindex(ts::TimeSeries, i::Int)

Returns the `i`th sample as `time => data`.
"""
function Base.getindex(ts::TimeSeries, i::Int)
    return ts.time[i] => ts.data[i]
end

"""
    getindex(ts::TimeSeries, i::Union{Colon, AbstractVector})

Returns a sliced `TimeSeries` that preserves metadata.
"""
function Base.getindex(ts::TimeSeries, i::Union{Colon, AbstractVector})
    return TimeSeries(;
        ts.title,
        time = ts.time[i],
        data = ts.data[i],
        ts.time_dimension,
        dimensions = copy(ts.dimensions),
        ts.path,
        ts.discrete,
        ts.interpolator,
        ts.groups,
    )
end

# This helper is shared by interpolation policies that need the usual "first sample at or
# after t" lookup. Keeping it outside `TimeSeries(t)` means interpolators that do not need
# this search do not have to pay for it.
function sample_index_at_or_before(ts::TimeSeries, t)
    if isempty(ts.time)
        error("Cannot evaluate an empty TimeSeries.")
    end
    t_first = first(ts.time)
    t_last = last(ts.time)
    if t < t_first || t > t_last
        error("Time $t is outside the range [$t_first, $t_last].")
    end
    return searchsortedlast(ts.time, t)
end

function (::SampleAndHold)(ts::TimeSeries, t)
    k = sample_index_at_or_before(ts, t)
    return ts.data[k]
end

function (::LinearInterpolation)(ts::TimeSeries, t)
    k_last = sample_index_at_or_before(ts, t) # should be a valid index or we'd have errored
    t_last = ts.time[k_last]
    y_last = ts.data[k_last]
    if t == t_last
        return y_last # This is what we were asked for.
    end
    k_next = k_last + 1
    t_next = ts.time[k_next]
    y_next = ts.data[k_next]
    fraction_from_last_to_next = (t - t_last) / (t_next - t_last)
    return y_last + fraction_from_last_to_next * (y_next - y_last)
end

"""
    ts(t)

Evaluates a `TimeSeries` at time `t`.

In-range access always delegates to `ts.interpolator`, which defaults to sample-and-hold
for discrete series and linear interpolation for ordinary continuous series.
"""
function (ts::TimeSeries)(t)
    return ts.interpolator(ts, t)
end

"""
    ts(times::AbstractVector)

Evaluates a `TimeSeries` at multiple time points and returns a new `TimeSeries`.
Each output sample is generated by calling `ts(t)` for the corresponding time.
"""
function (ts::TimeSeries)(times::AbstractVector)
    collected_times = collect(times)
    return TimeSeries(;
        ts.title,
        time = collected_times,
        data = [ts(t) for t in collected_times],
        ts.time_dimension,
        dimensions = copy(ts.dimensions),
        ts.path,
        ts.discrete,
        ts.interpolator,
        ts.groups,
    )
end

# These are for push!(ts, (t, x)).
function Base.push!(ts::TimeSeries, p::Pair)
    push!(ts, p.first, p.second)
end

# These are for push!(ts, t, x).
function Base.push!(::TimeSeries, t, ::Missing)
    # Ignore missing data.
end
function Base.push!(ts::TimeSeries, t, x)
    push!(ts.time, t)
    push!(ts.data, x)
end

# The 3-argument `Base.show` method is used by `display` for human-readable output.
function Base.show(io::IO, ::MIME"text/plain", ts::TimeSeries)
    println(io, "$(length(ts.time))-element TimeSeries of $(eltype(ts.data)) elements")
    println(io, "  title: $(ts.title)")
    if !isempty(ts.time)
        println(io, "  time: $(eltype(ts.time))[$(first(ts.time)), ... $(last(ts.time))]")
        println(io, "  data: $(eltype(ts.data))[$(first(ts.data)), ... $(last(ts.data))]")
    else
        println(io, "  time: $(eltype(ts.time))[]")
        println(io, "  data: $(eltype(ts.data))[]")
    end
    println(io, "  time_dimension: \"$(ts.time_dimension.label)\" => \"$(ts.time_dimension.units)\"")
    if isempty(ts.dimensions)
        println(io, "  dimensions: (none)")
    else
        println(io, "  dimensions:")
        for (k, dim) in enumerate(ts.dimensions)
            println(io, "    $k: \"$(dim.label)\" => \"$(dim.units)\"")
        end
    end
    println(io, "  path: $(ts.path)")
    println(io, "  discrete: $(ts.discrete)")
    println(io, "  interpolator: $(typeof(ts.interpolator))")
    if isempty(ts.groups)
        println(io, "  groups: (none)")
    else
        println(io, "  groups:")
        for (group_label, group_dim_labels) in ts.groups
            println(io, "    $group_label: $group_dim_labels")
        end
    end
end

# SystemsOfSystemsMakieExt picks this up:
function plot_ts(ts; kwargs...)
    error("There is no implementation of plot_ts. Import GLMakie (or any Makie package) to use plot_ts.")
end
function plot_ts!(f, ts; kwargs...)
    error("There is no implementation of plot_ts!. Import GLMakie (or any Makie package) to use plot_ts.")
end

end
