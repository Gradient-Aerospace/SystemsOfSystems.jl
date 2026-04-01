module SystemsOfSystemsMakieExt

using Makie: Figure, Axis, lines!, scatter!, Legend, Cycled, linkxaxes!
using Dimensions: getdim
using SystemsOfSystems: TimeSeries
import SystemsOfSystems

# This might be used by an empty slot in the "matrix" plot_ts implementation.
function SystemsOfSystems.plot_ts!(f, ts::Nothing)
    return Axis[]
end

"""
    plot_ts!(f, ts::TimeSeries)

Adds a TimeSeries to a given figure (or any "block" within a figure), `f`. All new axes
are returned.
"""
function SystemsOfSystems.plot_ts!(f, ts::TimeSeries)

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

    plot_fcn = ts.discrete ? scatter! : lines!

    # For each axis...
    axes = Axis[]
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
        push!(axes, a)

        # Plot each dimension called out in that group.
        for k in dim_nums
            label = if units_are_consistent
                ts.dimensions[k].label # Units are on the ylabel.
            else
                ts.dimensions[k].label * "(" * ts.dimensions[k].units * ")"
            end
            plot_fcn(a, t, [getdim(el, k) for el in data]; label)
        end

        # Add a legend.
        Legend(f[axis_num, 2], a)

    end

    return axes

end

"""
    plot_ts!(f, tss::Vector{<:Pair{String, <:TimeSeries}}; skip_units_check = false)

This combines multiple time series in a single plot in the given figure (or any "block"),
`f`. The `tss` input is a vector of string-time-series pairs, where the string becomes the
legend label for the plot. This ignores plot groups; every dimension gets its own axis.

By default, this checks to make sure the units are consistent and errors if they are not.
Set `skip_units_check = true` to skip the check.

All new axes are returned.

See `plot_ts(tss::Vector{<:Pair{String, <:TimeSeries}})` for more.
"""
function SystemsOfSystems.plot_ts!(f, tss::Vector{<:Pair{String, <:TimeSeries}}; skip_units_check = false)

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
    axes = [
        Axis(f[k, 1];
            xlabel = "$(ts1.time_dimension.label) ($(ts1.time_dimension.units))",
            title = k == 1 ? ts1.title : "",
            ylabel = "$(dim.label) ($(dim.units))",
        )
        for (k, dim) in enumerate(ts1.dimensions)
    ]

    # Now add the lines, with labels showing the thing they came from.
    for (ts_num, (label, ts)) in enumerate(tss)
        t = collect(ts.time)
        data = collect(ts.data)
        plot_fcn = ts.discrete ? scatter! : lines!
        for k in eachindex(ts.dimensions)
            plot_fcn(axes[k], t, [getdim(el, k) for el in data]; label, color = Cycled(ts_num))
        end
    end

    # Add a legend to each axis.
    for k in eachindex(ts1.dimensions)
        Legend(f[k, 2], axes[k])
    end

    return axes

end

"""
    plot_ts(ts::TimeSeries)

Plots all of the dimensions of a single `TimeSeries`, returning the Makie.Figure. Any
`figure_kwargs` will be passed to the Makie.Figure. If there is no content, `nothing` is
returned.
"""
function SystemsOfSystems.plot_ts(ts::TimeSeries; figure_kwargs = (;))
    f = Figure(; figure_kwargs...)
    axes = SystemsOfSystems.plot_ts!(f, ts)
    if isempty(axes)
        return nothing
    end
    linkxaxes!(axes)
    return f
end

"""
    plot_ts(tss::Vector{<:Pair{String, <:TimeSeries}}; skip_units_check = false)

This combines multiple time series in a single plot. The input is a vector of
string-time-series pairs, where the string becomes the legend label for the plot. This
ignores plot groups; every dimension gets its own axis.

By default, this checks to make sure the units are consistent and errors if they are not.
Set `skip_units_check = true` to skip the check.

If there is no content, `nothing` is returned.

Example:

```
plot_ts(
    [
        "truth" => truth_ts,
        "measured" => measured_ts,
    ]
)
```
"""
function SystemsOfSystems.plot_ts(tss::Vector{<:Pair{String, <:TimeSeries}}; skip_units_check = false, figure_kwargs = (;))
    f = Figure(; figure_kwargs...)
    axes = plot_ts!(f, tss; skip_units_check)
    if isempty(axes)
        return nothing
    end
    linkxaxes!(axes)
    return f
end

"""
    plot_ts(tss::Vector, figure_kwargs = (;))

This combines multiple time series in a single plot, stacked vertically. Any
`figure_kwargs` will be passed to the Makie.Figure. If there is no content, `nothing` is
returned.

Example:

```
plot_ts(truth_ts, measured_ts])
```
"""
function SystemsOfSystems.plot_ts(tss::Vector, figure_kwargs = (;))
    f = Figure(; figure_kwargs...)
    all_axes = Axis[]
    for (k, ts) in enumerate(tss)
        these_axes = SystemsOfSystems.plot_ts!(f[k, 1], ts)
        for axis in these_axes
            push!(all_axes, axis)
        end
    end
    if isempty(all_axes)
        return nothing
    end
    linkxaxes!(all_axes)
    return f
end

"""
    plot_ts(tss::Matrix, figure_kwargs = (;))

This combines multiple time series in a single plot, arranged in a matrix. Any
`figure_kwargs` will be passed to the Makie.Figure. If there is no content, `nothing` is
returned.

Example:

```
plot_ts(
    [
        ts1  ts2;
        ts3  ts4;
    ]
)
```
"""
function SystemsOfSystems.plot_ts(tss::Matrix; figure_kwargs = (;))
    f = Figure(; figure_kwargs...)
    (nr, nc) = size(tss)
    all_axes = Axis[]
    for r in 1:nr
        for c in 1:nc
            ts = tss[r, c]
            these_axes = SystemsOfSystems.plot_ts!(f[r, c], ts)
            for axis in these_axes
                push!(all_axes, axis)
            end
        end
    end
    if isempty(all_axes)
        return nothing
    end
    linkxaxes!(all_axes)
    return f
end

end
