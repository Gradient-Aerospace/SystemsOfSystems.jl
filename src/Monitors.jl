"""
TODO
"""
module Monitors

using ProgressMeter: Progress, update!, finish!

abstract type AbstractMonitorOptions end
abstract type AbstractMonitor end

using ..SystemsOfSystems: AbstractStopReason

###############
# ProgressBar #
###############

"""
TODO
"""
@kwdef struct ProgressBarOptions <: AbstractMonitorOptions
    update_interval::Float64 = 1.0
    description::String = "Simulating... "
end

struct ProgressBar <: AbstractMonitor
    progress::Progress
    t_start::Float64
end

# TODO: Should this take in the initial model?
function create_monitor(options::ProgressBarOptions, t_start, t_end) # Inputs?
    return ProgressBar(
        Progress(
            Int64(floor(1000 * (t_end - t_start)));
            dt = options.update_interval,
            desc = options.description,
        ),
        float(t_start),
    )
end

# TODO: Should this take in the model?
function update_monitor!(monitor::ProgressBar, t)
    update!(monitor.progress, Int64(floor(1000 * (t - monitor.t_start))))
end

# TODO: Should this take in the model?
function close_monitor!(monitor::ProgressBar, t)
    finish!(monitor.progress)
end

end
