"""
Model filters assign a logging sampler to each model path while constructing a simulation
log.

Each model receives one sampler. During simulation, that sampler controls the model's own
state and output logging and whether logging descends farther into the model tree.
"""
module ModelFilters

export AbstractModelFilter, get_model_sampler, AllPassModelFilter

using ..Samplers: AbstractSampler, CompleteSampler, NullSampler

"""
    AbstractModelFilter

The interface for assigning logging samplers to model paths.

Subtypes implement `get_model_sampler(model_filter, model_path)`, returning an
`AbstractSampler`. Paths use `/` for the root model and slash-separated names such as
`/car/drivetrain/engine` for submodels.

Filters select samplers during log construction; they do not currently prevent creation of
model histories or time-series containers. At runtime, a parent sampler that declines
submodel logging prevents descendant samplers from being consulted for that sample.
"""
abstract type AbstractModelFilter end

"""
    get_model_sampler(model_filter, model_path)

Return the `AbstractSampler` assigned by `model_filter` to `model_path`.

Custom `AbstractModelFilter` implementations must define this method for model paths passed
as strings.
"""
function get_model_sampler end

"""
    AllPassModelFilter()

Assign a `CompleteSampler` to every model. This is the default filter for RAM and HDF5 logs
and preserves complete logging behavior.
"""
@kwdef struct AllPassModelFilter <: AbstractModelFilter
end

get_model_sampler(::AllPassModelFilter, ::String) = CompleteSampler()

"""
    RegexModelEntry(; expression, sampler)
    RegexModelEntry(expression => sampler)

Associate an `expression::Regex` with a `sampler::AbstractSampler`.

When `expression` occurs in a model path, a `RegexModelFilter` can assign `sampler` to that
model. Use anchors when the match must cover a particular part of the path. An entry can be
written explicitly or constructed from an `expression => sampler` pair.
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
from the first entry whose regular expression occurs in the model path. Put specific rules
before broader fallbacks such as `r".*"`. Entries can be provided explicitly or as
`expression => sampler` pairs. A model that matches no entry receives a `NullSampler`.

Matching a model does not override its parent. A parent sampler must allow submodel logging
before this model's sampler will be consulted for a particular sample.

For example:

```
using SystemsOfSystems: ModelFilters, Samplers

ModelFilters.RegexModelFilter([
    r"^/\$" => Samplers.RegularSampler(1//10),
    r"^/" => Samplers.CompleteSampler(),
])
```

Here the root sampler gates the entire tree at 0.1 simulation-time intervals. Whenever the
root permits traversal, every descendant uses a `CompleteSampler` and logs normally.
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
