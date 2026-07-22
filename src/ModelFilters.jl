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
    RegexModelEntry(expression => sampler)

Contains an `expression::Regex` and `sampler::AbstractSampler`. If a model's path matches
the regular expression, then the given `sampler` will be used. An entry can also be
constructed from a pair containing a `Regex` and an `AbstractSampler`.
"""
@kwdef struct RegexModelEntry
    expression::Regex
    sampler::AbstractSampler
end

function RegexModelEntry(entry::Pair{Regex, <:AbstractSampler})
    return RegexModelEntry(;
        expression = entry.first,
        sampler = entry.second,
    )
end

function Base.convert(
    ::Type{RegexModelEntry},
    entry::Pair{Regex, <:AbstractSampler},
)
    return RegexModelEntry(entry)
end

"""
    RegexModelFilter(; entries)
    RegexModelFilter(entries)

Contains `entries`, a vector of `RegexModelEntry`. Each model will be given the sampler
corresponding with the first matched regular expression. In this way, it can be useful to
put specific model requirements earlier in the vector with fallbacks (like `r".*"`) later in
the list. Entries can be provided explicitly or as `expression => sampler` pairs. If a
model's path matches no entries, it will not be logged.

For example:

```
ModelFilters.RegexModelFilter([
    r"^/\$" => Samplers.RegularSampler(1//10),
    r"^/" => Samplers.CompleteSampler(),
])
```
"""
@kwdef struct RegexModelFilter <: AbstractModelFilter
    entries::Vector{RegexModelEntry}
end

function RegexModelFilter(entries::AbstractVector)
    return RegexModelFilter(RegexModelEntry[entry for entry in entries])
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
