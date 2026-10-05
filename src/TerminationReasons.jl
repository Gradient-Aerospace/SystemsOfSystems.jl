module TerminationReasons

export AbstractTerminationReason, AbstractStopReason, AbstractFailureReason,
    AbstractInterruption, finished, failed, interrupted, describe, details,
    UnknownStopReason, ReachedEndTime, ModelRequestedStop, HookRequestedStop, Interrupted,
    EncounteredError,
    export_termination_reason, TerminationSummary

using ..SystemsOfSystems: ExactTime, Hooks

# Abstract types

"""
The common supertype for every reason a simulation ceased running.

Normal stop requests and failures are deliberately separate categories. A model or hook
request is part of the modeled lifecycle; a numerical or software failure means that
lifecycle could not produce another valid sample.
"""
abstract type AbstractTerminationReason end

"""
A normal, successfully processed request to stop a simulation.
"""
abstract type AbstractStopReason <: AbstractTerminationReason end

"""
A condition that prevented the simulation from producing another valid accepted sample.
"""
abstract type AbstractFailureReason <: AbstractTerminationReason end

"""
Indicates that the simulation was interrupted, such as by a sigint resulting from ctrl+c.
"""
abstract type AbstractInterruption <:  AbstractTerminationReason end

# Termination reason interface

"""
    finished(stop::AbstractTerminationReason)

Returns true if the simulation propagated to a nominal end state, like reaching the
specified end time or to a model-requested termination.
"""
finished(stop::AbstractTerminationReason) = stop isa AbstractStopReason

"""
    failed(stop::AbstractTerminationReason)

Returns true if the simulation failed to propagate, such as experiencing a numerical
failure or exception.
"""
failed(stop::AbstractTerminationReason) = stop isa AbstractFailureReason

"""
Returns true if the simulation was interrupted, such as a sigint, in which case the
simulation did not continue to a nominal conclusion but also did not fail.
"""
interrupted(stop::AbstractTerminationReason) = stop isa AbstractInterruption

"""
    describe(reason::AbstractTerminationReason)

Returns a concise, human-readable description of why a simulation stopped.
"""
describe(reason::AbstractTerminationReason) = string(typeof(reason))

"""
    details(reason::AbstractTerminationReason)

Returns a string containing detailed information about the termination reason.
"""
details(reason::AbstractTerminationReason) = ""

# Individual termination reasons

"""
Internal sentinel indicating that the simulation loop should continue.
"""
struct UnknownStopReason <: AbstractStopReason end
describe(::UnknownStopReason) = "The sim stopped for an unknown reason."

"""
The simulation successfully processed its requested final sample.
"""
struct ReachedEndTime <: AbstractStopReason
    t_end::ExactTime
end
describe(stop::ReachedEndTime) =
    "The sim reached the specified end time of $(float(stop.t_end))."

"""
The first model encountered in deterministic hierarchy order requested a normal stop.
"""
struct ModelRequestedStop <: AbstractStopReason
    model_path::String
    reason::String
end
describe(stop::ModelRequestedStop) =
    "A model ($(stop.model_path)) requested a stop: $(stop.reason)."

"""
The first hook encountered in configured order requested a normal stop.
"""
struct HookRequestedStop <: AbstractStopReason
    t::ExactTime
    hook::Hooks.AbstractHook
end
describe(stop::HookRequestedStop) =
    "A $(stop.hook) hook requested a stop at t = $(float(stop.t))."

"""
    Interrupted(t)

A catchable interruption stopped the simulation at its last fully accepted time, `t`.
This is a normal stop; `succeeded(history)` is true for an interrupted simulation.
"""
struct Interrupted <: AbstractInterruption
    t::ExactTime
end
describe(stop::Interrupted) = "The sim was interrupted at t = $(float(stop.t))."

"""
User model code or simulation infrastructure raised an unexpected exception.
"""
struct EncounteredError <: AbstractFailureReason
    time::Float64
    exception::Exception
    trace::Any
end
describe(::EncounteredError) = "The sim experienced an error."
details(stop::EncounteredError) = sprint(showerror, stop.exception, stop.trace)

# A portable termination reason that represents the full API.

"""
Stores the results of the complete API for an `AbstractTerminationReason` in a portable way.
This simple type is easy to save to and load from an HDF5 file or YAML file, etc.

Fields:

* `type::String`: The string representing the original termination reason's type
* `finished::Bool`: The result of calling `finished` on the original termination
* `failed::Bool`: The result of calling `failed` on the original termination
* `interrupted::Bool`: The result of calling `interrupted` on the original termination
* `summary::String`: The result of calling `describe` on the original termination
* `details::String`: The result of calling `details` on the original termination

All defaults are `""` or `false`.

See `export_termination_reason`.
"""
@kwdef struct TerminationSummary <: AbstractTerminationReason
    type::String = ""
    finished::Bool = false
    failed::Bool = false
    interrupted::Bool = false
    summary::String = ""
    details::String = ""
end

# Fill in the complete API for termination reasons:
finished(stop::TerminationSummary) = stop.finished
failed(stop::TerminationSummary) = stop.failed
interrupted(stop::TerminationSummary) = stop.interrupted
describe(stop::TerminationSummary) = stop.summary
details(stop::TerminationSummary) = stop.details

"""
    export_termination_reason(stop::AbstractTerminationReason)

Returns a `TerminationSummary` for the given termination reason.
"""
function export_termination_reason(stop::AbstractTerminationReason)
    return TerminationSummary(;
        type = string(typeof(stop)),
        finished = finished(stop),
        failed = failed(stop),
        interrupted = interrupted(stop),
        summary = describe(stop),
        details = details(stop),
    )
end

# If it's already a summary, just keep it.
function export_termination_reason(stop::TerminationSummary)
    return stop
end

end
