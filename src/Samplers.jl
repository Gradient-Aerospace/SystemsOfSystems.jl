module Samplers

export get_logging_directive, should_log_states, should_log_outputs, should_log_models

using ..SimulationTimes: ExactTime

"""
    AbstractSampler

Subtypes are expected to implement `get_logging_directive(t, sampler)`, returning a type
that supports `should_log_states`, `should_log_outputs`, and `should_log_models`.
"""
abstract type AbstractSampler end

"""
    LoggingDirective

A type for easily describing what to log for a given model.
"""
@kwdef struct LoggingDirective <: AbstractSampler
    log_states::Bool
    log_outputs::Bool
    log_models::Bool
end

@inline should_log_states(x::LoggingDirective) = x.log_states
@inline should_log_outputs(x::LoggingDirective) = x.log_outputs
@inline should_log_models(x::LoggingDirective) = x.log_models

# This allows the LoggingDirective, itself, to fulfill the AbstractSampler interface.
@inline get_logging_directive(::ExactTime, x::LoggingDirective) = x

"""
    CompleteSampler

Logs all states, outputs, and models on all samples.
"""
@kwdef struct CompleteSampler <: AbstractSampler
end

@inline get_logging_directive(::ExactTime, x::CompleteSampler) = x
@inline should_log_states(x::CompleteSampler) = true
@inline should_log_outputs(x::CompleteSampler) = true
@inline should_log_models(x::CompleteSampler) = true

"""
    NullSampler

Logs no states, outputs, or models on any sample.
"""
@kwdef struct NullSampler <: AbstractSampler
end

@inline get_logging_directive(::ExactTime, x::NullSampler) = x
@inline should_log_states(x::NullSampler) = false
@inline should_log_outputs(x::NullSampler) = false
@inline should_log_models(x::NullSampler) = false

"""
    RegularSampler

Subsamples logs at the given `period`, only accepting times that are exact multiples of the
period.
"""
@kwdef struct RegularSampler <: AbstractSampler
    period::ExactTime
end

@inline function get_logging_directive(t::ExactTime, sampler::RegularSampler)
    if isinteger(t / sampler.period)
        return LoggingDirective(;
            log_states = true,
            log_outputs = true,
            log_models = true,
        )
    else
        return LoggingDirective(;
            log_states = false,
            log_outputs = false,
            log_models = false,
        )
    end
end

end
