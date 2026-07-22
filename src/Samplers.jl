"""
Logging samplers decide which parts of a model should be recorded at each accepted
simulation time.

A sampler receives the exact simulation time through `get_sampling_directive` and returns a
directive that independently controls state logging, output logging, and traversal into
submodels. Built-in samplers cover complete, absent, and regular-period logging; users can
extend `AbstractSampler` to implement other policies.
"""
module Samplers

export get_sampling_directive, should_log_states, should_log_outputs, should_log_models

using ..SimulationTimes: ExactTime, exact_time
using ..Schedules: is_regular_step_triggering

"""
    AbstractSampler

The interface for model-history sampling policies.

Subtypes implement `get_sampling_directive(t, sampler)`. The returned directive must support
`should_log_states`, `should_log_outputs`, and `should_log_models`. It may be a
`SamplingDirective`, the sampler itself, or another suitable type.

Samplers are consulted for every logging opportunity, including the initial discrete states
and outputs at the simulation start. Implementations should be deterministic and safe to
query more than once at the same exact time.
"""
abstract type AbstractSampler end

"""
    get_sampling_directive(t, sampler)

Return the sampling instructions for `sampler` at exact simulation time `t`.

Custom `AbstractSampler` implementations must define this method. The returned value is
queried with `should_log_states`, `should_log_outputs`, and `should_log_models`.
"""
function get_sampling_directive end

"""
    should_log_states(directive)

Return whether the model's continuous and discrete states should be logged for this sample.
"""
function should_log_states end

"""
    should_log_outputs(directive)

Return whether the model's continuous and discrete outputs should be logged for this
sample. Only outputs supplied by the corresponding rates or updates result are recorded.
"""
function should_log_outputs end

"""
    should_log_models(directive)

Return whether logging should continue recursively into this model's submodels for this
sample. Returning `false` prevents every descendant sampler from being consulted at that
time.
"""
function should_log_models end

"""
    SamplingDirective(; log_states, log_outputs, log_models)

A logging instruction with independent controls for a model's states, outputs, and
submodels.

`SamplingDirective` also implements the `AbstractSampler` interface by returning itself from
`get_sampling_directive`. It can therefore be used directly as a model's fixed sampler.
"""
@kwdef struct SamplingDirective <: AbstractSampler
    log_states::Bool
    log_outputs::Bool
    log_models::Bool
end

@inline should_log_states(x::SamplingDirective) = x.log_states
@inline should_log_outputs(x::SamplingDirective) = x.log_outputs
@inline should_log_models(x::SamplingDirective) = x.log_models

# This allows the SamplingDirective, itself, to fulfill the AbstractSampler interface.
@inline get_sampling_directive(::ExactTime, x::SamplingDirective) = x

"""
    CompleteSampler()

Log the model's states and outputs and continue into its submodels at every simulation-loop
sample.
"""
@kwdef struct CompleteSampler <: AbstractSampler
end

@inline get_sampling_directive(::ExactTime, x::CompleteSampler) = x
@inline should_log_states(x::CompleteSampler) = true
@inline should_log_outputs(x::CompleteSampler) = true
@inline should_log_models(x::CompleteSampler) = true

"""
    NullSampler()

Skip the model's states and outputs and do not continue into its submodels for every
simulation-loop sample.

The model history and any time-series containers selected by its model logging policy are
still created during initialization.
"""
@kwdef struct NullSampler <: AbstractSampler
end

@inline get_sampling_directive(::ExactTime, x::NullSampler) = x
@inline should_log_states(x::NullSampler) = false
@inline should_log_outputs(x::NullSampler) = false
@inline should_log_models(x::NullSampler) = false

"""
    RegularSampler(; period, offset = 0, continue_to_submodels = false)
    RegularSampler(period, offset = 0, continue_to_submodels = false)

Log the model's states and outputs and continue into its submodels at times in the sequence
`offset + n * period`, for nonnegative integer `n`. At other times, this model's states and
outputs are skipped, and `continue_to_submodels` controls whether descendant samplers are
still consulted.

`period` and `offset` are converted to exact simulation times. `period` must be finite and
strictly positive, and `offset` must be finite. A sampler does not add times to the
simulation scheduler: it only selects from accepted simulation times that already exist.
"""
struct RegularSampler <: AbstractSampler

    period::ExactTime
    offset::ExactTime
    continue_to_submodels::Bool

    function RegularSampler(period, offset = 0, continue_to_submodels::Bool = false)

        period = exact_time(period)
        offset = exact_time(offset)
        isfinite(period) || throw(ArgumentError("period must be finite."))
        isfinite(offset) || throw(ArgumentError("offset must be finite."))
        period > 0 || throw(ArgumentError("period must be positive."))
        return new(period, offset, continue_to_submodels)

    end

end
RegularSampler(; period, offset = 0, continue_to_submodels = false) =
    RegularSampler(period, offset, continue_to_submodels)

@inline function get_sampling_directive(t::ExactTime, sampler::RegularSampler)
    if is_regular_step_triggering(t, sampler.period, sampler.offset)
        return SamplingDirective(;
            log_states = true,
            log_outputs = true,
            log_models = true,
        )
    else
        return SamplingDirective(;
            log_states = false,
            log_outputs = false,
            log_models = sampler.continue_to_submodels,
        )
    end
end

end
