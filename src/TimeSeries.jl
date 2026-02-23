"""
TODO
"""
module TimeSeriesStuff

export Dimension, TimeSeries, plot_ts

"""
TODO
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
@kwdef struct TimeSeries{TVT, DVT}
    title::String
    time::TVT
    data::DVT
    time_dimension::Dimension
    dimensions::Vector{Dimension} # TODO: Consider "data_dimensions" for consistency.
    path::String # TODO: Consider "ID" instead of path. How is this even used?
    discrete::Bool = false
end

Base.getindex(ts::TimeSeries, i::Int) = ts.data[i]
function Base.getindex(ts::TimeSeries, i::Union{Colon, AbstractVector})
    return TimeSeries(
        ts.title,
        ts.time[i],
        ts.data[i],
        ts.time_dimension,
        copy(ts.dimensions),
        ts.path,
        ts.discrete,
    )
end

function (ts::TimeSeries)(t)
    if isempty(ts.time)
        error("Cannot evaluate an empty TimeSeries.")
    end
    t_first = first(ts.time)
    t_last = last(ts.time)
    if t < t_first || t > t_last
        error("Time $t is outside the range [$t_first, $t_last].")
    end

    k_hi = searchsortedfirst(ts.time, t)
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

    if ts.discrete
        return ts.data[k_hi - 1]
    end

    t_lo = ts.time[k_hi - 1]
    if t_hi == t_lo
        error("Cannot linearly interpolate when consecutive time points are identical at t = $t_lo.")
    end
    y_lo = ts.data[k_hi - 1]
    y_hi = ts.data[k_hi]
    α = (t - t_lo) / (t_hi - t_lo)
    return y_lo + α * (y_hi - y_lo)
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
end

# SystemsOfSystemsMakieExt picks this up:
function plot_ts(ts)
    error("There is no implementation of plot_ts. Import GLMakie (or any Makie package) to use plot_ts.")
end

end
