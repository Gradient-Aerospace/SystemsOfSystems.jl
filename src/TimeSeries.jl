"""
TODO
"""
module TimeSeriesStuff

export Dimension, TimeSeries, plot_ts

using Dimensions: numdims_for_type

"""
    Dimension(; label = "", units = "")

Used to label a dimension of a time series.
"""
struct Dimension
    label::String
    units::String
end
Dimension(; label = "", units = "") = Dimension(label, units)
Base.convert(::Type{Dimension}, pair::Pair) = Dimension(pair.first, pair.second)

"""
TODO
"""
struct TimeSeries{TVT, DVT}
    title::String
    time::TVT
    data::DVT
    time_dimension::Dimension
    dimensions::Vector{Dimension} # TODO: Consider "data_dimensions" for consistency.
    path::String # TODO: Consider "ID" instead of path. How is this even used?
    discrete::Bool
    groups::Vector{Pair{String, Vector{String}}}
end
# TODO: Test that the dimensions in the group are all available as part of the data.

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

    # Make sure all dimension labels are unique.
    labels = [dim.label for dim in dimensions]
    for label in labels
        if count(==(label), labels) > 1
            @error "Dimension labels must be unique, but the $label label was used multiple times in the $title time series (path = $path)."
        end
    end

    # Make sure the groups are valid.
    for (group_label, group_dimension_labels) in groups

        # Make sure there are no degenerate groups.
        @assert !isempty(group_dimension_labels) "A dimension group was empty for the $title time series (path = $path), but this is not allowed."

        # Make sure all of the dimensions in the dimension groups reference valid labels.
        for dim_label in group_dimension_labels
            if count(==(dim_label), labels) != 1
                @error "The $group_label dimension group of the $title time series (path = $path) referenced a dimension labeled $dim_label, but there is no dimension with that label. Valid labels: $labels."
            end
        end

    end

    return TimeSeries{typeof(time), typeof(data)}(
        title, time, data, time_dimension, dimensions, path, discrete, groups,
    )

end

"""
    getindex(ts::TimeSeries, i::Int)

Return the `i`th sample as `time => data`.
"""
function Base.getindex(ts::TimeSeries, i::Int)
    return ts.time[i] => ts.data[i]
end

"""
    getindex(ts::TimeSeries, i::Union{Colon, AbstractVector})

Return a sliced `TimeSeries` that preserves metadata.
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
        groups = [], # TODO: What to do about groups?
    )
end

"""
    ts(t)

Evaluate a `TimeSeries` at time `t`.

For continuous series (`discrete == false`), this uses linear interpolation.
For discrete series (`discrete == true`), this uses zero-order hold.
"""
function (ts::TimeSeries)(t)

    if isempty(ts.time)
        error("Cannot evaluate an empty TimeSeries.")
    end
    t_first = first(ts.time)
    t_last = last(ts.time)
    if t < t_first || t > t_last
        error("Time $t is outside the range [$t_first, $t_last].")
    end

    # Find the index of the first sample at or after t.
    k_hi = searchsortedfirst(ts.time, t)

    # Keep endpoint behavior explicit.
    if k_hi == 1
        return ts.data[1]
    end
    if k_hi > length(ts.time)
        return ts.data[end]
    end

    t_hi = ts.time[k_hi]
    if t == t_hi
        return ts.data[k_hi]
    end

    # Discrete time series are interpreted as zero-order hold.
    if ts.discrete
        return ts.data[k_hi - 1]
    end

    # Continuous time series use linear interpolation.
    t_lo = ts.time[k_hi - 1]
    if t_hi == t_lo
        error("Cannot linearly interpolate when consecutive time points are identical at t = $t_lo.")
    end
    y_lo = ts.data[k_hi - 1]
    y_hi = ts.data[k_hi]
    α = (t - t_lo) / (t_hi - t_lo)
    return y_lo + α * (y_hi - y_lo)

end

"""
    ts(times::AbstractVector)

Evaluate a `TimeSeries` at multiple time points and return a new `TimeSeries`.
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
        groups = [], # TODO: What to do about groups?
    )
end

# TODO: Should this be push!(ts, (t, x)) to follow the push!(coll, el) pattern?
function Base.push!(ts::TimeSeries, t, x::Missing)
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
function plot_ts(ts)
    error("There is no implementation of plot_ts. Import GLMakie (or any Makie package) to use plot_ts.")
end

end
