"""
    LoggingPolicies

This module contains types and functions for describing what to log in the simulation and
when to log it.

At the top level, many log types (BasicLogOptions, HDF5LogOptions) have a field for a
_logging policy_. This policy is responsible for assigning a _model logging policy_ to each
model, according the model's path (e.g., "/car/drivetrain/engine"). The model logging policy
describes which variables (which states, outputs, and submodels) are to be logged, as well
as a sampling rule for when those things are to be logged.

Here is a simple example:

```
log = BasicLogOptions(;
    logging_policy = AllPassModelLoggingPolicy(),
)
```

This logs all variables of all models on all samples and is, of course, the default.

Here's an example that controls which models are logged, what variables are logged, and when
they are logged:

```
log = BasicLogOptions(;
    logging_policy = RegexLoggingPolicy(
        [
            # Sample the root model every 0.1s.
            r"^/\$" => ModelLoggingPolicy(;
                sampler = Samplers.RegularSampler(;
                    period = 1//10,
                    continue_to_submodels = false,
                ),
            ),
            # Sample the children of some_model more coarsely (1s).
            r"^/some_model/" => ModelLoggingPolicy(;
                sampler = Samplers.RegularSampler(;
                    period = 1//1,
                ),
            ),
            # This one model has some variables that we don't want logged.
            r"^/path/to/some_weird_model\$" => ModelLoggingPolicy(;
                sampler = CompleteSampler(), # Log all samples of this model.
                variable_set = VariableExclusionList( # But don't log these variables.
                    [
                        "my_pointer",
                        "my_enormous_state_variable",
                    ]
                ),
            ),
        ],
        default = AllPassModelLoggingPolicy(),
    ),
)
```

Let's break that apart. The `RegexLoggingPolicy` uses regular expressions to map model paths
to model logging policies. The first "hit" in the vector is used. In this case, the first
entry is `r"^/\$"` which will only match exactly `"/"`. Therefore, the root model will
sample at 10Hz. Further, we've set `continue_to_submodels` to `false` (which is also the
default), so after logging this model, logging will not continue to submodels. It therefore
governs the rate of all models in the simulation.

The next item matches any model path starting with `"/some_model/"`, so all of the children
of `"/some_model"`, and samples them more slowly. (Note that sampleing them faster would
have no effect since the top-level model only samples at 10Hz and has
`continue_to_submodels = false`.)

The next item matches exactly one model by complete name, and it excludes certain variables
from logging.

If a model path matches none of the entries, then the default is used.
"""
module LoggingPolicies

using ..Samplers: AbstractSampler, CompleteSampler, NullSampler

#################
# Variable Sets #
#################

export AbstractVariableSet, AllVariables, VariableList, VariableExclusionList,
    is_variable_in_set

"""
    AbstractVariableSet

Subtypes control which variables of a model should be logged. Subtypes should implement
`is_variable_in_set`.
"""
abstract type AbstractVariableSet end

"""
    AllVariables()

Logs all of the model's variables.
"""
@kwdef struct AllVariables <: AbstractVariableSet end
is_variable_in_set(variable::Symbol, ::AllVariables) = true

"""
    NoVariables()

Logs none of the model's variables.
"""
@kwdef struct NoVariables <: AbstractVariableSet end
is_variable_in_set(variable::Symbol, ::NoVariables) = false

"""
    VariableList(; list::Vector{Symbol})

Logs all of the model's variables that are in the list.
"""
struct VariableList <: AbstractVariableSet
    list::Vector{Symbol}
end
VariableList(list::Vector{String}) = VariableList(symbol.(list))
VariableList(; list) = VariableList(list)
is_variable_in_set(variable::Symbol, set::VariableList) = in(variable, set.list)

"""
    VariableExclusionList(; list::Vector{Symbol})

Logs all of the model's variables except for those that are in the list.
"""
struct VariableExclusionList <: AbstractVariableSet
    list::Vector{Symbol}
end
VariableExclusionList(list::Vector{String}) = VariableExclusionList(symbol.(list))
VariableExclusionList(; list) = VariableExclusionList(list)
is_variable_in_set(variable::Symbol, set::VariableExclusionList) = !in(variable, set.list)

##########################
# Model Logging Policies #
##########################

export AbstractModelLoggingPolicy, get_sampler, get_variable_set,
    ModelLoggingPolicy, AllPassModelLoggingPolicy

"""
    AbstractModelLoggingPolicy

The interface for model logging policies. All subtypes are expected to implement
`get_sampler` and `get_variable_set`.
"""
abstract type AbstractModelLoggingPolicy end

"""
    get_sampler(::AbstractModelLoggingPolicy)

Returns an AbstractSampler for the given model logging policy.
"""
function get_sampler end

"""
    get_variable_set(::AbstractModelLoggingPolicy)

Returns an AbstractVariableSet for the given model logging policy.
"""
function get_variable_set end

"""
    AllPassModelLoggingPolicy()

A model logging policy that logs all of the model's variables on all samples.
"""
@kwdef struct AllPassModelLoggingPolicy <: AbstractModelLoggingPolicy
end
get_sampler(::AllPassModelLoggingPolicy) = CompleteSampler()
get_variable_set(::AllPassModelLoggingPolicy) = AllVariables()

"""
    NullModelLoggingPolicy()

A model logging policy that logs none of the model's variables, ever.
"""
@kwdef struct NullModelLoggingPolicy <: AbstractModelLoggingPolicy
end
get_sampler(::NullModelLoggingPolicy) = NullSampler()
get_variable_set(::NullModelLoggingPolicy) = NoVariables()

"""
    ModelLoggingPolicy(; sampler::AbstractSampler, variable_set::AbstractVariableSet)

A model logging policy that allows the user to explicitly set the `sampler` and
`variable_set` to use for a given model.
"""
@kwdef struct ModelLoggingPolicy <: AbstractModelLoggingPolicy
    sampler::AbstractSampler = CompleteSampler() # When to sample
    variable_set::AbstractVariableSet = AllVariables() # Which variables to log
end
get_sampler(policy::ModelLoggingPolicy) = policy.sampler
get_variable_set(policy::ModelLoggingPolicy) = policy.variable_set

####################
# Logging Policies #
####################

export AbstractLoggingPolicy, get_model_logging_policy,
    AllPassLoggingPolicy, RegexLoggingPolicyRule, RegexLoggingPolicy

"""
    AbstractLoggingPolicy

The interface for assigning model logging policies to individual models according to their
paths.

Subtypes implement `get_model_logging_policy(policy, model_path)`, returning an
`AbstractModelLoggingPolicy`. Paths use `"/"` for the root model and slash-separated names
such as `"/car/drivetrain/engine"` for submodels.
"""
abstract type AbstractLoggingPolicy end

"""
    get_model_logging_policy(policy::AbstractLoggingPolicy, model_path::String)

Return the `AbstractModelLoggingPolicy` assigned by `policy` to `model_path`.

Custom `AbstractLoggingPolicies` implementations must define this method.
"""
function get_model_logging_policy end

"""
    AllPassLoggingPolicy()

Assign an `AllPassModelLoggingPolicy` to every model. This is the default policy for
`BasicLog` and `HDF5Log`.
"""
@kwdef struct AllPassLoggingPolicy <: AbstractLoggingPolicy
end
get_model_logging_policy(::AllPassLoggingPolicy, ::String) = AllPassModelLoggingPolicy()

"""
    RegexLoggingPolicyRule(; expression::Regex, policy::AbstractModelLoggingPolicy)
    RegexLoggingPolicyRule(expression => policy)

Associates an `expression` with a model logging `policy`.
"""
struct RegexLoggingPolicyRule
    expression::Regex
    policy::AbstractModelLoggingPolicy
end

function RegexLoggingPolicyRule(
    expression::AbstractString,
    policy::AbstractModelLoggingPolicy,
)
    return RegexLoggingPolicyRule(Regex(expression), policy)
end

# Turns `r"..." => policy` into a RegexLoggingPolicyRule.
function RegexLoggingPolicyRule(rule::Pair)
    return RegexLoggingPolicyRule(rule.first, rule.second)
end

# Feeds a `convert` request to the above constructor.
function Base.convert(::Type{RegexLoggingPolicyRule}, rule::Pair)
    return RegexLoggingPolicyRule(rule)
end

"""
    RegexLoggingPolicy(; entries, default)
    RegexLoggingPolicy(entries, default)

Contains `entries`, a vector of `RegexLoggingPolicyRule`. Each model will be given the model
logging policy from the first entry whose regular expression occurs in the model path.
Entries can be provided as `RegexLoggingPolicyRule` or as `expression => sampler` pairs.
A model that matches no entry receives the `default` (which is `NullSampler` by default).

Example:

```
using SystemsOfSystems: LoggingPolicies, Samplers

LoggingPolicies.RegexLoggingPolicy(;
    entries = [
        r"^/my_model\$" => Samplers.RegularSampler(1//10),
        r"^/my_other_model/" => Samplers.RegularSampler(1//1),
    ],
    default = LoggingPolicies.AllPassModelLoggingPolicy(),
)
```

Here, `/my_model` and all models beneath it will be logged at 0.1s intervals, while all of
the children in `/my_other_model/` will be logged at 1s intervals, and all other models will
logged completely.
"""
struct RegexLoggingPolicy <: AbstractLoggingPolicy

    entries::Vector{RegexLoggingPolicyRule}
    default::AbstractModelLoggingPolicy

    function RegexLoggingPolicy(
        entries::Vector{RegexLoggingPolicyRule},
        default = NullModelLoggingPolicy(),
    )
        return new(entries, default)
    end
    function RegexLoggingPolicy(entries::AbstractVector, args...)
        return new(RegexLoggingPolicyRule[entry for entry in entries], args...)
    end
    RegexLoggingPolicy(; entries, default = NullModelLoggingPolicy()) = new(entries, default)

end
# function RegexLoggingPolicy(entries::AbstractVector, default = NullModelLoggingPolicy())
#     return RegexLoggingPolicy(RegexLoggingPolicyRule[entry for entry in entries], default)
# end
# function RegexLoggingPolicy(; entries, default = NullModelLoggingPolicy())
#     return RegexLoggingPolicy(entries, default)
# end

function get_model_logging_policy(policy::RegexLoggingPolicy, model_path::String)
    for entry in policy.entries
        if occursin(entry.expression, model_path)
            return entry.policy
        end
    end
    return policy.default
end

end
