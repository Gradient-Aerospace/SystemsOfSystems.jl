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

# TODO: Should this be push!(ts, (t, x)) to follow the push!(coll, el) pattern?
function Base.push!(ts::TimeSeries, t, x)
    push!(ts.time, t)
    push!(ts.data, x)
end

# The 3-argument `Base.show` method is used by `display` for human-readable output.
function Base.show(io::IO, ::MIME"text/plain", ts::TimeSeries)
    println(io, "$(length(ts.time))-element TimeSeries of $(eltype(ts.data)) elements")
    println(io, "  title: $(ts.title)")
    println(io, "  time: $(eltype(ts.time))[$(first(ts.time)), ... $(last(ts.time))]")
    println(io, "  data: $(eltype(ts.data))[$(first(ts.data)), ... $(last(ts.data))]")
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
