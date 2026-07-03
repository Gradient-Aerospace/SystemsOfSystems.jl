"""
    Hooks Module

Hooks provide a way to interact with the sim loop, such as updating a progress bar or
providing real-time synchronization.

See `AbstractHook` for more.
"""
module Hooks

using ProgressMeter: Progress, update!, finish!

"""
    AbstractHookOptions

An abstract type for a set of options used to construct a subtype of `AbstractHook`.
"""
abstract type AbstractHookOptions end

"""
    AbstractHook

An abstract type for functionality that "hooks into" the sim loop.

All subtypes are expected to provide the following interface:

* `create_hook`: Turns the hook's options (`AbstractHookOptions`) into the hook itself.
* `update_hook!`: Called at the beginning of each sim step, this allows a hook to update its
  internal state.
* `close_hook!`: Called at the end of the sim (whether the sim ended for nominal reasons or
  caught an error), allowing the hook to close any resources it's using, such as i/o.
"""
abstract type AbstractHook end

"""
    create_hook(options::AbstractHookOptions, t_start, t_end)

Returns a subtype of `AbstractHook` built from the provided `options`, where `t_start` and
`t_end` are the start and end times of the simulation.
"""
function create_hook end

"""
    update_hook!(hook::AbstractHook, t)

Allows the hook to update its internal state at time `t`.
"""
function update_hook! end

"""
    close_hook!(hook::AbstractHook, t)

Called at the end of the simulation (whether the sim completed nominally or had an error),
allowing the hook to close i/o resources, summarize, etc.
"""
function close_hook! end

###############
# ProgressBar #
###############

"""
    ProgressBarOptions

Stores the options for a command-line progress bar, including `update_interval` for how
often the progress bar should update (seconds) and `description` to use for the progress
bar's text.
"""
@kwdef struct ProgressBarOptions <: AbstractHookOptions
    update_interval::Float64 = 1.0
    description::String = "Simulating... "
end

"""
    ProgressBar

See `ProgressBarOptions`.
"""
struct ProgressBar <: AbstractHook
    progress::Progress
    t_start::Float64
end

# TODO: Should this take in the initial model?
function create_hook(options::ProgressBarOptions, t_start, t_end) # Inputs?
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
function update_hook!(hook::ProgressBar, t)
    update!(hook.progress, Int64(floor(1000 * (t - hook.t_start))))
end

# TODO: Should this take in the model?
function close_hook!(hook::ProgressBar, t)
    finish!(hook.progress)
end

end
