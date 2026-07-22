"""
    LoggingPolicies

This module contains types and functions for describing what to log in the simulation and
when to log it.

At the top level, many log types (`BasicLogOptions`, `HDF5LogOptions`) have a field for a
_logging policy_. This policy is responsible for assigning a _model logging policy_ to each
model, according to the model's path (e.g., "/car/drivetrain/engine"). The model logging
policy describes which constants, states, and outputs are to be stored, as well as a
sampling rule for when the states and outputs are to be recorded.

Here is a simple example:

```
using SystemsOfSystems: LoggingPolicies, Logs

log = Logs.BasicLogOptions(;
    logging_policy = LoggingPolicies.AllPassLoggingPolicy(),
)
```

This logs all variables of all models on all samples and is, of course, the default.

Here's an example that controls which models are logged, what variables are logged, and when
they are logged:

```
using SystemsOfSystems: LoggingPolicies, Logs, Samplers

log = Logs.BasicLogOptions(;
    logging_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            # Sample the root model every 0.1s.
            r"^/\$" => LoggingPolicies.ModelLoggingPolicy(;
                sampler = Samplers.RegularSampler(;
                    period = 1//10,
                    continue_to_submodels = false,
                ),
            ),
            # Sample the children of some_model more coarsely (1s).
            r"^/some_model/" => LoggingPolicies.ModelLoggingPolicy(;
                sampler = Samplers.RegularSampler(;
                    period = 1//1,
                ),
            ),
            # This one model has some variables that we don't want logged.
            r"^/path/to/some_weird_model\$" => LoggingPolicies.ModelLoggingPolicy(;
                sampler = Samplers.CompleteSampler(), # Log all samples of this model.
                variable_set = LoggingPolicies.VariableExclusionList(
                    [
                        "my_pointer", # But don't log these variables.
                        "my_enormous_state_variable",
                    ]
                ),
            ),
        ],
        default = LoggingPolicies.AllPassModelLoggingPolicy(),
    ),
)
```

Let's break that apart. The `RegexLoggingPolicy` uses regular expressions to map model paths
to model logging policies. The first "hit" in the vector is used. In this case, the first
rule is `r"^/\$"`, which will only match exactly `"/"`. Therefore, the root model will
sample at 10 Hz (or rather, it will sample at all simulation times that align with a 0.1s
step). At other times, `continue_to_submodels = false` prevents logging from continuing to
submodels, so the root sampler also limits when any descendant can log.

The next item matches any model path starting with `"/some_model/"`, so all of the children
of `"/some_model"`, and samples them more slowly. (Note that sampling them faster would
have no effect since the top-level model only samples at 10 Hz and has
`continue_to_submodels = false`.)

The next item matches exactly one model by complete name, and it excludes certain variables
from logging.

If a model path matches none of the rules, then the default is used.

Logging policies do not alter the simulation scheduler. A regular sampler records only the
matching times among samples already accepted because of solver steps, model schedules,
model requests, or user-provided times.
"""
module LoggingPolicies

using ..Samplers: AbstractSampler, CompleteSampler, NullSampler

#################
# Variable Sets #
#################

export AbstractVariableSet, AllVariables, NoVariables, VariableList, VariableExclusionList,
    is_variable_in_set

"""
    AbstractVariableSet

Subtypes control which variables of a model should be logged. Subtypes should implement
`is_variable_in_set`. Variable sets apply to constants, states, and outputs; submodel
histories are assigned their own model logging policies.
"""
abstract type AbstractVariableSet end

"""
    is_variable_in_set(variable::Symbol, set::AbstractVariableSet)

Return whether the named constant, state, or output should be stored for a model assigned
`set`. Custom variable sets implement this method.
"""
function is_variable_in_set end

"""
    AllVariables()

Select all of the model's constants, states, and outputs.
"""
@kwdef struct AllVariables <: AbstractVariableSet end
is_variable_in_set(variable::Symbol, ::AllVariables) = true

"""
    NoVariables()

Select none of the model's constants, states, or outputs.
"""
@kwdef struct NoVariables <: AbstractVariableSet end
is_variable_in_set(variable::Symbol, ::NoVariables) = false

"""
    VariableList(list)
    VariableList(; list)

Select the model variables whose names are in `list`. Names may be strings or symbols.
"""
struct VariableList <: AbstractVariableSet
    list::Vector{Symbol}
end
VariableList(list::AbstractVector) = VariableList(Symbol.(list))
VariableList(; list) = VariableList(list)
is_variable_in_set(variable::Symbol, set::VariableList) = in(variable, set.list)

"""
    VariableExclusionList(list)
    VariableExclusionList(; list)

Select all model variables except those whose names are in `list`. Names may be strings or
symbols.
"""
struct VariableExclusionList <: AbstractVariableSet
    list::Vector{Symbol}
end
VariableExclusionList(list::AbstractVector) = VariableExclusionList(Symbol.(list))
VariableExclusionList(; list) = VariableExclusionList(list)
is_variable_in_set(variable::Symbol, set::VariableExclusionList) = !in(variable, set.list)

##########################
# Model Logging Policies #
##########################

export AbstractModelLoggingPolicy, get_sampler, get_variable_set,
    ModelLoggingPolicy, AllPassModelLoggingPolicy, NullModelLoggingPolicy

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

A model logging policy that stores all of the model's variables and records them at every
accepted sample.
"""
@kwdef struct AllPassModelLoggingPolicy <: AbstractModelLoggingPolicy
end
get_sampler(::AllPassModelLoggingPolicy) = CompleteSampler()
get_variable_set(::AllPassModelLoggingPolicy) = AllVariables()

"""
    NullModelLoggingPolicy()

A model logging policy that stores none of the model's variables and prevents logging from
traversing into its submodels. The model history itself is still present as a structural
node in the log.
"""
@kwdef struct NullModelLoggingPolicy <: AbstractModelLoggingPolicy
end
get_sampler(::NullModelLoggingPolicy) = NullSampler()
get_variable_set(::NullModelLoggingPolicy) = NoVariables()

"""
    ModelLoggingPolicy(; sampler::AbstractSampler, variable_set::AbstractVariableSet)

A model logging policy that explicitly sets which variables are stored with `variable_set`
and when stored states and outputs are recorded with `sampler`.
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
    AllPassLoggingPolicy, UniformLoggingPolicy, RegexLoggingPolicyRule, RegexLoggingPolicy

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

Custom `AbstractLoggingPolicy` implementations must define this method.
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
    UniformLoggingPolicy(policy::AbstractModelLoggingPolicy)
    UniformLoggingPolicy(; policy::AbstractModelLoggingPolicy)

Assign the same model logging policy to every model.
"""
@kwdef struct UniformLoggingPolicy <: AbstractLoggingPolicy
    policy::AbstractModelLoggingPolicy
end
get_model_logging_policy(policy::UniformLoggingPolicy, ::String) = policy.policy

"""
    RegexLoggingPolicyRule(; expression, policy)
    RegexLoggingPolicyRule(expression, policy)
    RegexLoggingPolicyRule(expression => policy)

Associate a regular `expression` with a model logging `policy`. A string expression is
compiled to a `Regex`. An `AbstractSampler` may be supplied as shorthand for a
`ModelLoggingPolicy` that selects all variables.
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

function RegexLoggingPolicyRule(expression, sampler::AbstractSampler)
    return RegexLoggingPolicyRule(expression, ModelLoggingPolicy(; sampler))
end

RegexLoggingPolicyRule(; expression, policy) = RegexLoggingPolicyRule(expression, policy)

# Turns `r"..." => policy` into a RegexLoggingPolicyRule.
function RegexLoggingPolicyRule(rule::Pair)
    return RegexLoggingPolicyRule(rule.first, rule.second)
end

# Feeds a `convert` request to the above constructor.
function Base.convert(::Type{RegexLoggingPolicyRule}, rule::Pair)
    return RegexLoggingPolicyRule(rule)
end

"""
    RegexLoggingPolicy(; rules, default)
    RegexLoggingPolicy(rules, default)

Contains `rules`, a vector of `RegexLoggingPolicyRule`. Each model will be given the model
logging policy from the first entry whose regular expression occurs in the model path.
Rules can be provided as `RegexLoggingPolicyRule`, `expression => policy`, or
`expression => sampler` pairs. A model that matches no rule receives `default`, which is a
`NullModelLoggingPolicy` by default. The first matching rule wins.

Example:

```
using SystemsOfSystems: LoggingPolicies, Samplers

LoggingPolicies.RegexLoggingPolicy(;
    rules = [
        r"^/my_model\$" => Samplers.RegularSampler(1//10),
        r"^/my_other_model/" => Samplers.RegularSampler(1//1),
    ],
    default = LoggingPolicies.AllPassModelLoggingPolicy(),
)
```

Here, exactly `/my_model` will be logged at 0.1 s intervals, while descendants of
`/my_other_model` will be logged at 1 s intervals, and all other models will be logged
completely.
"""
struct RegexLoggingPolicy <: AbstractLoggingPolicy
    rules::Vector{RegexLoggingPolicyRule}
    default::AbstractModelLoggingPolicy
end

function RegexLoggingPolicy(
    rules::AbstractVector,
    default::AbstractModelLoggingPolicy = NullModelLoggingPolicy(),
)
    return RegexLoggingPolicy(RegexLoggingPolicyRule[rule for rule in rules], default)
end
RegexLoggingPolicy(; rules, default = NullModelLoggingPolicy()) =
    RegexLoggingPolicy(rules, default)

function get_model_logging_policy(policy::RegexLoggingPolicy, model_path::String)
    for rule in policy.rules
        if occursin(rule.expression, model_path)
            return rule.policy
        end
    end
    return policy.default
end

end
