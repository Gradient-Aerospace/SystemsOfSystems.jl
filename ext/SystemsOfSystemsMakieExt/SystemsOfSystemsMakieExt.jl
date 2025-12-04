module SystemsOfSystemsMakieExt

using Makie: Figure, Axis, lines!, stairs!, Legend
using Dimensions: getdim
using SystemsOfSystems: TimeSeries
import SystemsOfSystems

"""
TODO
"""
function SystemsOfSystems.plot_ts(ts::TimeSeries)
    t = collect(ts.time)
    data = collect(ts.data)
    f = Figure()
    plot_fcn = ts.discrete ? stairs! : lines!
    for (k, dim) in enumerate(ts.dimensions)
        a = Axis(f[k, 1];
            xlabel = "$(ts.time_dimension.label) ($(ts.time_dimension.units))",
            title = k == 1 ? ts.title : "",
            ylabel = "$(dim.label) ($(dim.units))",
        )
        plot_fcn(a, t, [getdim(el, k) for el in data]; label = dim.label)
    end
    return f
end

"""
TODO
"""
function SystemsOfSystems.plot_ts(tss::Vector{<:Pair{String, <:TimeSeries}}; skip_units_check = false)

    ts1 = first(tss)[2]

    # We can only combine these plots if they have the same dimensions, so check for that.
    nd = length(ts1.dimensions)
    if !all(nd == length(ts.dimensions) for (_, ts) in tss)
        error("The time series cannot be combined into a single plot; their data sets have different dimensionality.")
    end
    if !skip_units_check
        if !all(ts1.dimensions[d].units == ts.dimensions[d].units for (_, ts) in tss for d in 1:nd)
            error("The time series cannot be combined into a single plot; their data sets have different units.")
        end
    end

    # Make the axes using the first time series.
    f = Figure()
    a = [
        Axis(f[k, 1];
            xlabel = "$(ts1.time_dimension.label) ($(ts1.time_dimension.units))",
            title = k == 1 ? ts1.title : nothing,
            ylabel = "$(dim.label) ($(dim.units))",
        )
        for (k, dim) in enumerate(ts1.dimensions)
    ]

    # Now add the lines, with labels showing the thing they came from.
    for (label, ts) in tss
        t = collect(ts.time)
        data = collect(ts.data)
        for k in eachindex(ts.dimensions)
            lines!(a[k], t, [getdim(el, k) for el in data]; label)
        end
    end

    # Add a legend to each axis.
    for k in eachindex(ts1.dimensions)
        Legend(f[k, 2], a[k])
    end

    return f

end

end
