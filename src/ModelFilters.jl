module ModelFilters

export AbstractModelFilter, get_model_sampler, AllPassModelFilter

using ..Samplers: AbstractSampler, CompleteSampler, NullSampler

"""
    AbstractModelFilter

All subtypes are expected to implement `get_model_sampler(model_filter, model_path)`,
returning an `AbstractSampler`.
"""
abstract type AbstractModelFilter end

"""
    AllPassModelFilter

Allows all models to pass, giving each a `CompleteSampler` (logs everything).
"""
@kwdef struct AllPassModelFilter <: AbstractModelFilter
end

get_model_sampler(::AllPassModelFilter, ::String) = CompleteSampler()

"""
    RegexModelEntry

Contains an `expression::Regex` and `sampler::AbstractSampler`. If a model's path matches
the regular expression, then the given `sampler` will be used.
"""
@kwdef struct RegexModelEntry
    expression::Regex
    sampler::AbstractSampler
end

"""
    RegexModelFilter

Contains `entries`, a vector of `RegexModelEntry`. Each model will be given the sampler
corresponding with the first matched regular expression. In this way, it can be useful to
put specific model requirements earlier in the vector with fall-backs (like r".*") latter in
the list. If a model's path matches no entries, it will not be logged.
"""
@kwdef struct RegexModelFilter <: AbstractModelFilter
    entries::Vector{RegexModelEntry}
end

function get_model_sampler(filter::RegexModelFilter, model_path::String)
    for entry in filter.entries
        if occursin(entry.expression, model_path)
            return entry.sampler
        end
    end
    return NullSampler()
end

end
