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
    create_hook(options::AbstractHookOptions, t, model)

Returns a subtype of `AbstractHook` built from the provided `options`, where `t` is an array
or `Rational{Int64}` corresponding to the set of times passed to `simulate` (i.e.,
`first(t)` is when the sim will start, `last(t)` is when it will end, and anything in
between is a desired output time for the sim, and `model` is the initial model.
"""
function create_hook end

"""
    update_hook!(hook::AbstractHook, t, model)

Allows the `hook` to update its internal state at time `t` using the `model`.
"""
function update_hook! end

"""
    close_hook!(hook::AbstractHook, t_end, model)

Called at the end of the simulation (whether the sim completed nominally or had an error),
allowing the `hook` to close i/o resources, summarize, etc., where `t_end` is the final sim
time and `model` is the final model.
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

function create_hook(options::ProgressBarOptions, t, model)
    return ProgressBar(
        Progress(
            Int64(floor(1000 * (last(t) - first(t)))); # Make 1000 update "stages".
            dt = options.update_interval,
            desc = options.description,
        ),
        float(first(t)),
    )
end

function update_hook!(hook::ProgressBar, t, model)
    update!(hook.progress, Int64(floor(1000 * (t - hook.t_start))))
end

function close_hook!(hook::ProgressBar, t, model)
    finish!(hook.progress)
end

end
