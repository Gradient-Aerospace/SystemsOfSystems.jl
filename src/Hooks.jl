"""
    Hooks Module

Hooks provide a way to interact with the sim loop, such as updating a progress bar or
providing real-time synchronization.

See `AbstractHook` for more.
"""
module Hooks

using ProgressMeter: Progress, update!, finish!
using ..SimulationTimes: ExactTime, float_duration

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
    HookOutputs

A container for the output of `update_hook!`, allowing the hook to communicate with the
simulation loop.

Fields:

* `stop::Bool`: Set to true to stop the sim (default: false)
"""
@kwdef struct HookOutputs
    stop::Bool = false
end

"""
    create_hook(options::AbstractHookOptions, t, model)

Returns a subtype of `AbstractHook` built from the provided `options`, where `t` is an array
of exact simulation times corresponding to the set of times passed to `simulate` (i.e.,
`first(t)` is when the sim will start, `last(t)` is when it will end, and anything in
between is a desired output time for the sim, and `model` is the initial model.
"""
function create_hook end

"""
    update_hook!(hook::AbstractHook, t, model)

Allows the `hook` to update its internal state at time `t` using the `model`. The model will
correspond with continuous-time updates up to `t`, and it will not yet have performed its
discrete update at `t`.
"""
function update_hook! end

"""
    close_hook!(hook::AbstractHook, t_end, model)

Called at the end of the simulation (whether the sim completed nominally or had an error),
allowing the `hook` to close i/o resources, summarize, etc., where `t_end` is the final sim
time and `model` is the final model.
"""
function close_hook!(hook::AbstractHook, t_end, model)
    return nothing
end

###############
# ProgressBar #
###############

"""
    ProgressBarOptions

A container for command-line progress bar options, including `update_interval` for how
often the progress bar updates (seconds) and `description` for the progress bar's text.
"""
@kwdef struct ProgressBarOptions <: AbstractHookOptions
    update_interval::Float64 = 1.0
    description::String = "Simulating... "
end

"""
    ProgressBar

A runtime command-line progress-bar hook created from `ProgressBarOptions`.
"""
struct ProgressBar <: AbstractHook
    progress::Progress
    t_start::ExactTime
end

function create_hook(options::ProgressBarOptions, t, model)
    return ProgressBar(
        Progress(
            Int64(floor(1000 * float_duration(first(t), last(t))));
            dt = options.update_interval,
            desc = options.description,
        ),
        first(t),
    )
end

function update_hook!(hook::ProgressBar, t, model)
    elapsed = float_duration(hook.t_start, t)
    update!(hook.progress, Int64(floor(1000 * elapsed)))
    return HookOutputs()
end

function close_hook!(hook::ProgressBar, t, model)
    finish!(hook.progress)
    return nothing
end

##############
# SimTimeout #
##############

"""
    SimTimeoutOptions

A container for `SimTimeout` options, which can end a simulation that takes too long.

Fields:

* `max_run_time`: The maximum run time before the hook should terminate the sim (s)
"""
@kwdef struct SimTimeoutOptions <: AbstractHookOptions
    max_run_time::Float64
end

"""
    SimTimeout

A runtime simulation-timeout hook created from `SimTimeoutOptions`.
"""
@kwdef struct SimTimeout <: AbstractHook
    max_run_time_ns::UInt64
    initial_time_ns::UInt64
end

function create_hook(options::SimTimeoutOptions, t, model)
    return SimTimeout(;
        max_run_time_ns = UInt64(floor(options.max_run_time * 1e9)),
        initial_time_ns = time_ns(),
    )
end

function update_hook!(hook::SimTimeout, t, model)
    current_time_ns = time_ns()
    return HookOutputs(;
        stop = current_time_ns - hook.initial_time_ns > hook.max_run_time_ns,
    )
end

#############
# ClockSync #
#############

"""
    ClockSyncOptions

A container for `ClockSync` options, which keep the simulation loop from running faster
than real time. If the amount of desired stall time is larger than
`sleep_margin` (s), it will sleep until `sleep_margin` before the next trigger time. After
that, it enters a tight loop using `time_ns()` to determine when it's time to continue with
the simulation.

Since `time_ns()` is used for timing, it is unaffected by system clock updates, and it
updates continuously. This value ultimately comes from the operating system, the computer's
oscillator, and time synchronization sources, which are only used to determine how many
oscillations occur per externally-referenced unit of time. On Linux and macOS, the external
corrections prevent most drift. Its timing performance can vary by target platform,
especially for long-running simulations with high precision requirements.

Julia's `sleep` function has a minimum duration of 1ms. The default `sleep_margin` is 2ms to
allow the model to enter the "tight timing loop" after sleeping.

This type uses `UInt64` to store the start and current times in nanoseconds. This means that
real-time synchronization can be sustained for approximately 584 years and is unlikely to
limit the duration of the simulation.
"""
@kwdef struct ClockSyncOptions <: AbstractHookOptions
    sleep_margin::Float64 = 0.002
end

"""
    ClockSync

A runtime wall-clock synchronization hook created from `ClockSyncOptions`.
"""
@kwdef struct ClockSync <: AbstractHook
    t_start::ExactTime
    initial_time_ns::UInt64
    sleep_margin_ns::UInt64
end

function create_hook(options::ClockSyncOptions, t, model)
    return ClockSync(;
        t_start         = first(t),
        initial_time_ns = time_ns(),
        sleep_margin_ns = UInt64(floor(options.sleep_margin * 1e9)),
    )
end

function update_hook!(hook::ClockSync, t, model)

    # Figure what sim step we're up to.
    sim_time_ns = UInt64(floor(float_duration(hook.t_start, t) * 1e9))

    # See if we have time (and margin) to go to sleep.
    run_time_ns = time_ns() - hook.initial_time_ns
    if sim_time_ns > run_time_ns + hook.sleep_margin_ns
        sleep((sim_time_ns - (run_time_ns + hook.sleep_margin_ns)) * 1e-9)
    end

    # Now enter a tight loop until it's time to run.
    while (time_ns() - hook.initial_time_ns) < sim_time_ns
    end

    return HookOutputs()

end

end
