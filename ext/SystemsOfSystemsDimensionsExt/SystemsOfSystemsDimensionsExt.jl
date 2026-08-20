module SystemsOfSystemsDimensionsExt

import Dimensions
using SystemsOfSystems: TimeSeries, select

"""
    getdim(ts::TimeSeries, d)

Returns dimension `d` of each data element as a new `TimeSeries`.
"""
function Dimensions.getdim(ts::TimeSeries, d)

    return select(
        x -> Dimensions.getdim(x, d),
        ts;
        dimensions = [ts.dimensions[d],],
    )

end

end
