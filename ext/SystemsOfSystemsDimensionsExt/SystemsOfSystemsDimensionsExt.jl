module SystemsOfSystemsDimensionsExt

import Dimensions
using SystemsOfSystems: TimeSeries, select

"""
    getdim(ts::TimeSeries, k::Integer; kwargs)

Returns dimension `k` of each data element as a new `TimeSeries`.
Accepts the same metadata keyword arguments as `select`.
"""
function Dimensions.getdim(
    ts::TimeSeries,
    k::Integer;
    kwargs...,
)

    return select(
        x -> Dimensions.getdim(x, k),
        ts;
        title = ts.title * ", dimension = $k",
        dimensions = [ts.dimensions[k],],
        path = ts.path,
        kwargs...,
    )

end

end
