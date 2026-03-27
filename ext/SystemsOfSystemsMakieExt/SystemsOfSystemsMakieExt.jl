module SystemsOfSystemsMakieExt

using Makie: Figure, Axis, lines!, scatter!, Legend
using Dimensions: getdim
using SystemsOfSystems: TimeSeries
import SystemsOfSystems

"""
    plot_ts(ts::TimeSeries)

Plots all of the dimensions of a single `TimeSeries`, returning the Makie.Figure.
"""
function SystemsOfSystems.plot_ts(ts::TimeSeries)

    # If there are no plot groups, then there's nothing to plot.
    if isempty(ts.groups)
        return nothing
    end

    # Pull the set of dimensions. This is the only way we can build a figure. If those
    # aren't provided for some reason, bail out.
    dim_labels = [dim.label for dim in ts.dimensions]
    if isempty(dim_labels)
        return nothing
    end

    # Pull the data. We "collect" in case the data isn't in RAM (e.g., and HDF5Vector).
    t = collect(ts.time)
    data = collect(ts.data)

    f = Figure()
    plot_fcn = ts.discrete ? scatter! : lines!

    # For each axis...
    for (axis_num, (group_label, group_dimension_labels)) in enumerate(ts.groups)

        # This should not be possible because the TimeSeries itself checks for this, but we
        # may as well check.
        @assert !isempty(group_dimension_labels) "No dimension group should be empty."

        # Get the dimension numbers from the dimension labels.
        dim_nums = map(group_dimension_labels) do label
            for k in eachindex(dim_labels)
                if dim_labels[k] == label
                    return k
                end
            end
            @error "Could not find dimension $label. Valid labels: $dim_labels"
        end

        # We'll want to know if units are consistent to determine if they should be part of
        # the y label or part of the legend.
        first_units = ts.dimensions[dim_nums[1]].units
        units_are_consistent = all(
            ts.dimensions[k].units == first_units
            for k in dim_nums
        )
        ylabel = if units_are_consistent
            "$group_label ($first_units)"
        else
            group_label
        end

        # Get the axis started. If it's on top, add a title for the whole figure.
        a = Axis(f[axis_num, 1];
            title = axis_num == 1 ? ts.title : "",
            xlabel = "$(ts.time_dimension.label) ($(ts.time_dimension.units))",
            ylabel,
        )

        # Plot each dimension called out in that group.
        for k in dim_nums
            label = if units_are_consistent
                ts.dimensions[k].label # Units are on the ylabel.
            else
                ts.dimensions[k].label * "(" * ts.dimensions[k].units * ")"
            end
            plot_fcn(a, t, [getdim(el, k) for el in data]; label)
        end

    end

    return f

end

"""
TODO

This combines multiple time series in a single plot. The input is a vector of
string-time-series pairs, where the string becomes the legend label for the plot.
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

# Plot all of the dimensions of all of the time series stacked vertically.
function SystemsOfSystems.plot_ts(tss::Vector{<:TimeSeries})
    f = Figure()
    count = 0
    for ts in tss
        t = collect(ts.time)
        data = collect(ts.data)
        plot_fcn = ts.discrete ? scatter! : lines!
        for (k, dim) in enumerate(ts.dimensions)
            a = Axis(f[count + k, 1];
                xlabel = "$(ts.time_dimension.label) ($(ts.time_dimension.units))",
                title = k == 1 ? ts.title : "",
                ylabel = "$(dim.label) ($(dim.units))",
            )
            plot_fcn(a, t, [getdim(el, k) for el in data]; label = dim.label)
        end
        count += length(ts.dimensions)
    end
    return f
end

end
