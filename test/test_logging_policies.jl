module TestLoggingPolicies

using Test
import HDF5Vectors # Load the HDF5 extension.
using SystemsOfSystems
using SystemsOfSystems: LoggingPolicies, Logs, Samplers, Solvers

# Logging configuration has three layers that are useful to keep separate while reading
# these tests:
#
# 1. A logging policy assigns a model logging policy based on the model's path.
# 2. A model logging policy chooses which variables get storage and supplies a sampler.
# 3. A sampler decides which of the simulation's already-accepted times are recorded for
#    that model. Every submodel receives and follows its own sampler independently.
#
# The unit tests below exercise each layer on its own. The integration tests then use one
# deliberately small two-level model to show how the layers interact during a simulation.

# Every variable category in the synthetic model has one "keep" and one "drop" member.
# Reusing this list lets one assertion helper verify that filtering treats constants,
# continuous/discrete states, and continuous/discrete outputs consistently.
const SELECTED_VARIABLES = [
    :keep_constant,
    :keep_continuous_state,
    :keep_discrete_state,
    :keep_continuous_output,
    :keep_discrete_output,
]

# This custom sampler records how often the compiled logging plan queries it. Assigning one
# instance to multiple models lets the integration tests distinguish shared trigger
# evaluation from merely producing the same histories through repeated per-model calls.
mutable struct CountingSampler <: Samplers.AbstractSampler
    times::Vector{Rational{Int64}}
end
CountingSampler() = CountingSampler(Rational{Int64}[])

function Samplers.get_sampling_directive(t, sampler::CountingSampler)
    push!(sampler.times, t)
    return Samplers.SamplingDirective(true, true)
end

function model_description(; include_child = true)

    # A root and one child are deep enough to test path assignment and parent-to-child
    # traversal without obscuring the expected histories. The child omits another child so
    # this recursion terminates after two model levels.
    models = if include_child
        (; child = model_description(; include_child = false))
    else
        (;)
    end

    # Each category contains values with matching shapes and types. That makes any missing
    # or extra logged field attributable to the logging policy, rather than to differences
    # in how a particular model variable is represented.
    return ModelDescription(;
        constants = (;
            keep_constant = 1.0,
            drop_constant = -1.0,
        ),
        continuous_states = (;
            keep_continuous_state = 0.0,
            drop_continuous_state = 0.0,
        ),
        discrete_states = (;
            keep_discrete_state = 0,
            drop_discrete_state = 0,
        ),
        continuous_outputs = (;
            keep_continuous_output = 0.0,
            drop_continuous_output = 0.0,
        ),
        discrete_outputs = (;
            keep_discrete_output = 0.0,
            drop_discrete_output = 0.0,
        ),
        models,
    )

end

function rates(t, model)

    # RatesOutput must mirror the model hierarchy. Recursing here ensures that root and
    # child have real continuous samples, allowing traversal behavior to be observed in
    # their time vectors.
    models = if hasproperty(model, :child)
        (; child = rates(t, model.child))
    else
        (;)
    end

    # Simple opposite-valued signals make the keep/drop variables distinguishable while
    # keeping the numerical model irrelevant to these logging tests.
    return RatesOutput(;
        rates = (;
            keep_continuous_state = 1.0,
            drop_continuous_state = -1.0,
        ),
        outputs = (;
            keep_continuous_output = t,
            drop_continuous_output = -t,
        ),
        models,
    )

end

function updates(t, model)

    # As with rates, discrete updates are supplied for the complete hierarchy at every
    # accepted step. A missing child UpdatesOutput would test update propagation instead of
    # logging traversal, which is outside this file's purpose.
    models = if hasproperty(model, :child)
        (; child = updates(t, model.child))
    else
        (;)
    end

    # Both discrete states change on every accepted step, and both outputs are always
    # supplied. Consequently, gaps in a resulting history come only from its sampler.
    return UpdatesOutput(;
        updates = (;
            keep_discrete_state = model.keep_discrete_state + 1,
            drop_discrete_state = model.drop_discrete_state - 1,
        ),
        outputs = (;
            keep_discrete_output = float(t),
            drop_discrete_output = -float(t),
        ),
        models,
    )

end

function run_logging_simulation(logging_policy; log_options = nothing)

    # Most tests want the fast in-memory log. Accepting an already-constructed option keeps
    # the exact same model and solver available to the HDF5 tests below.
    if isnothing(log_options)
        log_options = Logs.BasicLogOptions(; logging_policy)
    end

    # RK4 with a quarter-second step gives the known accepted-time grid
    # [0, 0.25, 0.5, 0.75, 1]. Regular samplers can therefore be checked with exact expected
    # vectors instead of depending on the adaptive solver's chosen steps.
    history = simulate(
        nothing;
        init_fcn = (args...) -> model_description(),
        rates_fcn = rates,
        updates_fcn = updates,
        t = (0, 1),
        options = SimOptions(;
            log = log_options,
            solver = Solvers.RungeKutta4Options(; dt = 1//4),
        ),
    )
    return history

end

function sparse_update_model_description(; include_child = true)

    # This hierarchy reproduces the failure mode that a complete update tree in the primary
    # fixture cannot expose. Both models change once between regular logging samples and
    # then disappear from later UpdatesOutput trees.
    models = if include_child
        (; child = sparse_update_model_description(; include_child = false))
    else
        (;)
    end

    return ModelDescription(;
        discrete_states = (;
            sampled_state = 0,
        ),
        discrete_outputs = (;
            sampled_output = -1,
        ),
        models,
    )

end

function sparse_update_rates(t, model)

    # The simulation still needs a structurally complete RatesOutput tree even though this
    # fixture has no continuous variables. Keeping the continuous side empty ensures every
    # resulting sample comes from discrete logging.
    models = if hasproperty(model, :child)
        (; child = sparse_update_rates(t, model.child))
    else
        (;)
    end

    return RatesOutput(; models)

end

function sparse_updates(t, model)

    # At the first selected time after the change, provide a parent UpdatesOutput whose
    # child explicitly returns `nothing`. This exercises the nested no-event path while the
    # separate snapshot traversal records both models' current states.
    if t == 1//2 && hasproperty(model, :child)
        return UpdatesOutput(; models = (; child = nothing,))
    end

    # Emit exactly one state change and output event, halfway between the regular sampler's
    # selected times. All other opportunities return `nothing`, including the final selected
    # time. This verifies that no event is required for a regular state snapshot.
    if t != 1//4
        return nothing
    end

    models = if hasproperty(model, :child)
        (; child = sparse_updates(t, model.child))
    else
        (;)
    end
    return UpdatesOutput(;
        updates = (;
            sampled_state = 1,
        ),
        outputs = (;
            sampled_output = 1,
        ),
        models,
    )

end

function run_sparse_update_simulation(logging_policy; log_options = nothing)

    # Accepting either log implementation lets the same behavioral assertions exercise the
    # in-memory and direct-to-HDF5 paths.
    if isnothing(log_options)
        log_options = Logs.BasicLogOptions(; logging_policy)
    end

    history = simulate(
        nothing;
        init_fcn = (args...) -> sparse_update_model_description(),
        rates_fcn = sparse_update_rates,
        updates_fcn = sparse_updates,
        t = (0, 1),
        options = SimOptions(;
            log = log_options,
            solver = Solvers.RungeKutta4Options(; dt = 1//4),
        ),
    )
    return history

end

function assert_regular_state_snapshots(root_history)

    # A state exists at every selected time, including times when the current UpdatesOutput
    # omits it. The off-grid output is ephemeral, so only its initialization value remains.
    for model_history in (root_history, root_history["child"])
        @test collect(model_history["sampled_state"].time) == [0.0, 0.5, 1.0]
        @test collect(model_history["sampled_state"].data) == [0, 1, 1]
        @test collect(model_history["sampled_output"].time) == [0.0]
        @test collect(model_history["sampled_output"].data) == [-1]
    end

end

function missing_output_model_description()

    # Declaring Missing as part of the continuous output type deliberately verifies that
    # the value remains the no-sample sentinel rather than becoming logged data.
    return ModelDescription(;
        continuous_outputs = (;
            continuous_signal = VariableDescription{Union{Missing, Float64}}(
                missing;
                title = "Continuous Signal",
                dimensions = [],
            ),
        ),
        discrete_outputs = (;
            discrete_signal = VariableDescription{Float64}(
                missing;
                title = "Discrete Signal",
                dimensions = [],
            ),
        ),
    )

end

function run_missing_output_simulation(log_options)

    # Both output functions decline the half-second sample. Fixed quarter-second steps make
    # the expected retained timestamps independent of adaptive solver behavior.
    return simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> missing_output_model_description(),
        rates_fcn = (t, model) -> RatesOutput(;
            outputs = (;
                continuous_signal = t == 0.5 ? missing : t,
            ),
        ),
        updates_fcn = (t, model) -> UpdatesOutput(;
            outputs = (;
                discrete_signal = t == 1//2 ? missing : float(t),
            ),
        ),
        options = SimOptions(;
            log = log_options,
            solver = Solvers.RungeKutta4Options(; dt = 1//4),
        ),
    )

end

function assert_missing_outputs_are_skipped(model_history)

    continuous_signal = model_history["continuous_signal"]
    discrete_signal = model_history["discrete_signal"]
    @test collect(continuous_signal.time) == [0., 0.25, 0.75, 1.]
    @test collect(continuous_signal.data) == [0., 0.25, 0.75, 1.]
    @test collect(discrete_signal.time) == [0.25, 0.75, 1.]
    @test collect(discrete_signal.data) == [0.25, 0.75, 1.]
    @test all(!ismissing, continuous_signal.data)
    @test all(!ismissing, discrete_signal.data)

end

function assert_selected_variables(model_history)

    # This helper intentionally inspects the five storage categories directly. Checking
    # only ModelHistory.keys would not reveal a variable accidentally placed in the wrong
    # category, and repeating these assertions for RAM and HDF5 logs would be noisy.
    @test keys(model_history.constants) == (:keep_constant,)
    @test keys(model_history.continuous_states) == (:keep_continuous_state,)
    @test keys(model_history.discrete_states) == (:keep_discrete_state,)
    @test keys(model_history.continuous_outputs) == (:keep_continuous_output,)
    @test keys(model_history.discrete_outputs) == (:keep_discrete_output,)

end

@testset "variable sets" begin

    # Variable sets are pure membership rules used during log construction. Test them
    # independently first so later integration failures can be attributed to storage
    # creation rather than to the membership predicates themselves.
    all_variables = LoggingPolicies.AllVariables()
    no_variables = LoggingPolicies.NoVariables()
    variable_list = LoggingPolicies.VariableList(["one", "two"])
    exclusion_list = LoggingPolicies.VariableExclusionList(; list = [:one, :two])

    # The all/none sets are the building blocks for the corresponding model policies.
    @test LoggingPolicies.is_variable_in_set(:anything, all_variables)
    @test !LoggingPolicies.is_variable_in_set(:anything, no_variables)

    # Inclusion and exclusion lists should be complementary for names both inside and
    # outside the configured list.
    @test LoggingPolicies.is_variable_in_set(:one, variable_list)
    @test !LoggingPolicies.is_variable_in_set(:three, variable_list)
    @test !LoggingPolicies.is_variable_in_set(:one, exclusion_list)
    @test LoggingPolicies.is_variable_in_set(:three, exclusion_list)

    # String input is user-friendly, but the stored form is normalized to symbols so it
    # matches NamedTuple field names without conversion in the logging setup path.
    @test variable_list.list == [:one, :two]

end

@testset "sampling directives and built-in samplers" begin

    # CompleteSampler and NullSampler are constant directives at opposite extremes. A
    # SamplingDirective itself is also a sampler, which is useful when states and outputs
    # need different fixed answers.
    complete = Samplers.get_sampling_directive(0//1, Samplers.CompleteSampler())
    null = Samplers.get_sampling_directive(0//1, Samplers.NullSampler())
    states_only = Samplers.SamplingDirective(;
        log_states = true,
        log_outputs = false,
    )

    # Verify the state and output decisions independently.
    @test Samplers.should_log_states(complete)
    @test !Samplers.should_snapshot_states(complete)
    @test Samplers.should_log_outputs(complete)
    @test !Samplers.should_log_states(null)
    @test !Samplers.should_snapshot_states(null)
    @test !Samplers.should_log_outputs(null)
    @test Samplers.get_sampling_directive(0//1, states_only) === states_only
    @test Samplers.SamplingDirective(true, false) == states_only
    @test Samplers.should_log_states(states_only)
    @test !Samplers.should_snapshot_states(states_only)
    @test !Samplers.should_log_outputs(states_only)

    # Positional and keyword construction both convert ordinary real values to exact time.
    positional = Samplers.RegularSampler(0.5, 0.25)
    keyword = Samplers.RegularSampler(;
        period = 1//2,
        offset = 1//4,
    )
    @test positional.period == 1//2
    @test positional.offset == 1//4
    @test positional == keyword
    @test Samplers.RegularSampler(0.1).period == 1//10

    # The offset is the first sample. A model is inactive before and between its grid times.
    before_offset = Samplers.get_sampling_directive(0//1, keyword)
    on_grid = Samplers.get_sampling_directive(3//4, keyword)
    off_grid = Samplers.get_sampling_directive(1//2, keyword)
    @test !Samplers.should_log_states(before_offset)
    @test !Samplers.should_snapshot_states(before_offset)
    @test Samplers.should_log_states(on_grid)
    @test Samplers.should_snapshot_states(on_grid)
    @test Samplers.should_log_outputs(on_grid)
    @test !Samplers.should_log_states(off_grid)
    @test !Samplers.should_snapshot_states(off_grid)
    @test !Samplers.should_log_outputs(off_grid)

    # Invalid clocks are rejected during construction rather than failing in the loop.
    @test_throws ArgumentError Samplers.RegularSampler(0)
    @test_throws ArgumentError Samplers.RegularSampler(-1)
    @test_throws ArgumentError Samplers.RegularSampler(1//0)
    @test_throws ArgumentError Samplers.RegularSampler(1, 1//0)

end

@testset "model logging policies" begin

    # A model logging policy is only a composition of a storage selection and a sampler.
    # These interface tests ensure the convenience policies expand to the intended pair and
    # that an explicit policy preserves the exact objects supplied by the user.
    all_pass = LoggingPolicies.AllPassModelLoggingPolicy()
    null = LoggingPolicies.NullModelLoggingPolicy()
    variable_set = LoggingPolicies.VariableList(SELECTED_VARIABLES)
    sampler = Samplers.RegularSampler(1//2)
    explicit = LoggingPolicies.ModelLoggingPolicy(; sampler, variable_set)

    @test LoggingPolicies.get_sampler(all_pass) isa Samplers.CompleteSampler
    @test LoggingPolicies.get_variable_set(all_pass) isa LoggingPolicies.AllVariables
    @test LoggingPolicies.get_sampler(null) isa Samplers.NullSampler
    @test LoggingPolicies.get_variable_set(null) isa LoggingPolicies.NoVariables
    @test LoggingPolicies.get_sampler(explicit) === sampler
    @test LoggingPolicies.get_variable_set(explicit) === variable_set

end

@testset "logging-policy assignment" begin

    # This testset concerns the top-level path-to-policy mapping only. Simulation behavior
    # is tested later; here we can directly check conversion, path matching, precedence,
    # and fallback behavior without a model obscuring the result.
    root_sampler = Samplers.RegularSampler(1//2)
    child_policy = LoggingPolicies.ModelLoggingPolicy(;
        variable_set = LoggingPolicies.VariableList([:keep_constant]),
    )
    compact_rules = [
        r"^/$" => root_sampler,
        "/child\$" => child_policy, # Note that this is a string, not a Regex.
    ]

    # Pair conversion supports both sampler shorthand and explicit model policies.
    root_rule = convert(LoggingPolicies.RegexLoggingPolicyRule, first(compact_rules))
    child_rule = LoggingPolicies.RegexLoggingPolicyRule(last(compact_rules))
    @test root_rule.expression == r"^/$"
    @test LoggingPolicies.get_sampler(root_rule.policy) === root_sampler
    @test child_rule.expression == r"/child$"
    @test child_rule.policy === child_policy

    # Preserve the verbose construction form as well as the pair shorthand. The verbose
    # form is useful when rules are assembled or documented individually.
    explicit_rule = LoggingPolicies.RegexLoggingPolicyRule(;
        expression = r"^/explicit$",
        policy = child_policy,
    )
    @test explicit_rule.expression == r"^/explicit$"
    @test explicit_rule.policy === child_policy

    # Both constructors normalize compact rules, preserve first-match ordering, and safely
    # initialize the default policy.
    explicit_default = LoggingPolicies.AllPassModelLoggingPolicy()
    positional = LoggingPolicies.RegexLoggingPolicy(compact_rules, explicit_default)
    keyword = LoggingPolicies.RegexLoggingPolicy(;
        rules = compact_rules,
        default = explicit_default,
    )
    for policy in (positional, keyword)
        @test LoggingPolicies.get_sampler(
            LoggingPolicies.get_model_logging_policy(policy, "/"),
        ) === root_sampler
        @test LoggingPolicies.get_model_logging_policy(policy, "/child") === child_policy
        @test LoggingPolicies.get_model_logging_policy(policy, "/other") ===
            explicit_default
    end

    # An omitted default must be a fully initialized NullModelLoggingPolicy, rather than an
    # undefined struct field or a bare NullSampler.
    defaulted = LoggingPolicies.RegexLoggingPolicy(compact_rules)
    @test LoggingPolicies.get_model_logging_policy(defaulted, "/other") isa
        LoggingPolicies.NullModelLoggingPolicy

    # Both expressions match the root. The broad rule comes first deliberately to establish
    # that rule order, not specificity, determines the result.
    first_match = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => LoggingPolicies.NullModelLoggingPolicy(),
            r"^/$" => LoggingPolicies.AllPassModelLoggingPolicy(),
        ],
    )
    @test LoggingPolicies.get_model_logging_policy(first_match, "/") isa
        LoggingPolicies.NullModelLoggingPolicy

    # The non-regex policies cover the common cases where path matching is unnecessary.
    all_pass_policy = LoggingPolicies.AllPassLoggingPolicy()
    uniform_policy = LoggingPolicies.UniformLoggingPolicy(; policy = child_policy)
    @test LoggingPolicies.get_model_logging_policy(all_pass_policy, "/anything") isa
        LoggingPolicies.AllPassModelLoggingPolicy
    @test LoggingPolicies.get_model_logging_policy(uniform_policy, "/anything") ===
        child_policy

end

@testset "variable selection in model histories" begin

    # Apply a selective policy only to the root and an all-pass default to its child. This
    # demonstrates that storage decisions are made independently for each path during log
    # construction.
    root_policy = LoggingPolicies.ModelLoggingPolicy(;
        variable_set = LoggingPolicies.VariableList(string.(SELECTED_VARIABLES)),
    )
    policy = LoggingPolicies.RegexLoggingPolicy(
        [r"^/$" => root_policy],
        LoggingPolicies.AllPassModelLoggingPolicy(),
    )
    history = run_logging_simulation(policy)
    root_history = history["/"]
    child_history = history["/child"]

    # A variable set is applied consistently to constants and every state/output category.
    # The nonempty histories also confirm that selected containers remain connected to the
    # runtime logging functions, rather than merely being created during setup.
    assert_selected_variables(root_history)
    @test root_history["keep_constant"] == 1.0
    @test_throws ErrorException root_history["drop_constant"]
    @test !isempty(root_history["keep_continuous_state"].time)
    @test !isempty(root_history["keep_discrete_state"].time)
    @test !isempty(root_history["keep_continuous_output"].time)
    @test !isempty(root_history["keep_discrete_output"].time)

    # Model paths select policies independently; the explicitly all-pass child remains
    # complete even though its parent stores a subset of variables.
    @test length(keys(child_history.constants)) == 2
    @test length(keys(child_history.continuous_states)) == 2
    @test length(keys(child_history.discrete_states)) == 2
    @test length(keys(child_history.continuous_outputs)) == 2
    @test length(keys(child_history.discrete_outputs)) == 2

    # A null policy retains the hierarchy as structural model-history nodes without
    # allocating variable containers or recording samples. Retaining the nodes makes paths
    # and descendants navigable even when a model contributes no variables of its own.
    null_history = run_logging_simulation(LoggingPolicies.UniformLoggingPolicy(;
        policy = LoggingPolicies.NullModelLoggingPolicy(),
    ))
    for model_path in ("/", "/child")
        model_history = null_history[model_path]
        @test isempty(model_history.constants)
        @test isempty(model_history.continuous_states)
        @test isempty(model_history.discrete_states)
        @test isempty(model_history.continuous_outputs)
        @test isempty(model_history.discrete_outputs)
    end
    @test keys(null_history["/"].models) == (:child,)

end

@testset "independent model sampling and shared decisions" begin

    # Give the root an offset regular sampler and its child a complete sampler. The child's
    # history must remain complete when the root is off-grid, proving that a parent sampling
    # decision cannot suppress an independently assigned child.
    independent_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => Samplers.RegularSampler(1//2, 1//4),
            r"^/" => Samplers.CompleteSampler(),
        ],
    )
    independent_history = run_logging_simulation(independent_policy)
    @test independent_history["/"]["keep_continuous_state"].time == [0.25, 0.75]
    @test independent_history["/"]["keep_discrete_state"].time == [0.25, 0.75]
    @test independent_history["/"]["keep_discrete_state"].data == [1, 3]
    @test independent_history["/child"]["keep_continuous_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]
    @test independent_history["/child"]["keep_discrete_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]
    @test independent_history["/child"]["keep_discrete_state"].data == [0, 1, 2, 3, 4]
    @test !hasproperty(independent_history["/"], :sampler)
    @test !hasproperty(independent_history["/"], :sampling_group)
    @test !hasproperty(independent_history["/"], :sampling_groups_in_subtree)

    # NullModelLoggingPolicy used to stop traversal as well as suppressing its own model.
    # Under independent sampling, a null root must not hide a specifically enabled child.
    null_parent_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => LoggingPolicies.NullModelLoggingPolicy(),
            r"^/child$" => Samplers.CompleteSampler(),
        ],
    )
    null_parent_history = run_logging_simulation(null_parent_policy)
    @test isempty(null_parent_history["/"].continuous_states)
    @test isempty(null_parent_history["/"].discrete_states)
    @test null_parent_history["/child"]["keep_continuous_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]
    @test null_parent_history["/child"]["keep_discrete_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]

    # A broad rule is the common way to request one rate for the entire simulation. Both
    # models should use the same canonical sampling group so the regular trigger is tested
    # only once per logging opportunity and the whole tree can be rejected at its root.
    shared_sampler = Samplers.RegularSampler(1//2, 1//4)
    shared_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => shared_sampler,
        ],
    )
    shared_history = run_logging_simulation(shared_policy)
    shared_root = shared_history["/"]
    shared_child = shared_history["/child"]
    for model_history in (shared_root, shared_child)
        @test model_history["keep_continuous_state"].time == [0.25, 0.75]
        @test model_history["keep_continuous_output"].time == [0.25, 0.75]
        @test model_history["keep_discrete_state"].time == [0.25, 0.75]
        @test model_history["keep_discrete_state"].data == [1, 3]
        @test model_history["keep_discrete_output"].time == [0.25, 0.75]
    end
    # Sharing a group must also share the work, rather than simply sharing a sampler field.
    # This simulation has five accepted times. Each is a discrete logging opportunity and a
    # continuous logging opportunity, so the sampler should be called twice per time—not
    # once per time, phase, and model.
    counting_sampler = CountingSampler()
    counting_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => counting_sampler,
        ],
    )
    run_logging_simulation(counting_policy)
    @test length(counting_sampler.times) == 10
    for t in (0//1, 1//4, 1//2, 3//4, 1//1)
        @test count(==(t), counting_sampler.times) == 2
    end

end

@testset "regular state snapshots preserve off-grid changes" begin

    # Assign one regular sampler to the entire hierarchy. At every half-second time, both
    # models must snapshot their current states even though neither appears in that time's
    # UpdatesOutput tree.
    policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => Samplers.RegularSampler(1//2),
        ],
    )
    history = run_sparse_update_simulation(policy)
    assert_regular_state_snapshots(history["/"])

    # A regular sampler can also live below a complete parent. Its selected times must be
    # consulted even when the parent's current UpdatesOutput omits that child. The root
    # remains a sparse change-event history while the child forms regular snapshots.
    child_sampling_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => Samplers.CompleteSampler(),
            r"^/child$" => Samplers.RegularSampler(1//2),
        ],
    )
    child_sampling_history = run_sparse_update_simulation(child_sampling_policy)
    child_sampling_root = child_sampling_history["/"]
    @test child_sampling_root["sampled_state"].time == [0.0, 0.25]
    @test child_sampling_root["sampled_state"].data == [0, 1]
    @test child_sampling_root["child"]["sampled_state"].time == [0.0, 0.5, 1.0]
    @test child_sampling_root["child"]["sampled_state"].data == [0, 1, 1]

    # Without regular sampling, CompleteSampler retains the original sparse change-event
    # behavior. This guards against turning every accepted solver step into a redundant
    # discrete-state sample while fixing the regular-snapshot path.
    complete_history = run_sparse_update_simulation(
        LoggingPolicies.AllPassLoggingPolicy(),
    )
    for model_path in ("/", "/child")
        model_history = complete_history[model_path]
        @test model_history["sampled_state"].time == [0.0, 0.25]
        @test model_history["sampled_state"].data == [0, 1]
        @test model_history["sampled_output"].time == [0.0, 0.25]
        @test model_history["sampled_output"].data == [-1, 1]
    end

end

@testset "independent state and output directives" begin

    # RegularSampler selects states and outputs together, so use a fixed SamplingDirective
    # to prove the two flags are honored independently. The child uses its own all-pass
    # fallback even though root states are suppressed.
    output_only = Samplers.SamplingDirective(;
        log_states = false,
        log_outputs = true,
    )
    policy = LoggingPolicies.RegexLoggingPolicy(
        [r"^/$" => output_only],
        LoggingPolicies.AllPassModelLoggingPolicy(),
    )
    history = run_logging_simulation(policy)

    # Both continuous and discrete variants of each category must follow the same flag.
    @test isempty(history["/"]["keep_continuous_state"].time)
    @test isempty(history["/"]["keep_discrete_state"].time)
    @test !isempty(history["/"]["keep_continuous_output"].time)
    @test !isempty(history["/"]["keep_discrete_output"].time)
    @test !isempty(history["/child"]["keep_continuous_state"].time)
    @test !isempty(history["/child"]["keep_discrete_state"].time)

    # Snapshotting refines state logging rather than enabling it. A custom directive may
    # technically request a snapshot while disabling states; the compiled decision must
    # treat that combination as no state logging instead of letting the snapshot pass
    # bypass the primary flag.
    disabled_snapshot = Samplers.SamplingDirective(;
        log_states = false,
        snapshot_states = true,
        log_outputs = false,
    )
    disabled_snapshot_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => disabled_snapshot,
        ],
    )
    disabled_snapshot_history = run_logging_simulation(disabled_snapshot_policy)
    @test isempty(disabled_snapshot_history["/"]["keep_continuous_state"].time)
    @test isempty(disabled_snapshot_history["/"]["keep_discrete_state"].time)

end

@testset "missing output samples" begin

    # Missing always means that no output sample is available, even when the declared type
    # includes Missing. Verify the rule first with ordinary in-memory vectors.
    basic_history = run_missing_output_simulation(Logs.BasicLogOptions())
    assert_missing_outputs_are_skipped(basic_history["/"])

    # Direct-to-HDF5 histories use the same TimeSeries interface over different vector
    # storage. Check both the live history and the reloaded file so omitted samples cannot
    # be hidden by either storage layer.
    mktempdir() do directory

        filename = joinpath(directory, "missing_outputs.h5")
        history = run_missing_output_simulation(Logs.HDF5LogOptions(filename))
        assert_missing_outputs_are_skipped(history["/"])
        Logs.close_log(history.log)

        loaded_log, loaded_history = Logs.load_hdf5_log(filename)
        assert_missing_outputs_are_skipped(loaded_history)
        Logs.close_log(loaded_log)

    end

end

@testset "filtered HDF5 logs" begin

    # HDF5 uses the same ModelHistory interface as an in-memory log, but it also writes a
    # separate list of variable names as metadata. A filtered log is valid only if both the
    # time-series groups and that metadata describe precisely the same selected variables.
    selected_policy = LoggingPolicies.UniformLoggingPolicy(;
        policy = LoggingPolicies.ModelLoggingPolicy(;
            variable_set = LoggingPolicies.VariableList(SELECTED_VARIABLES),
        ),
    )

    # Use an isolated directory because HDF5 logs own open file handles and the test creates
    # both a direct-to-disk log and a saved copy of an in-memory log.
    mktempdir() do directory

        # Direct-to-HDF5 logging stores metadata for exactly the selected variables, so the
        # resulting file can be loaded without references to omitted time-series groups.
        filename = joinpath(directory, "filtered_log.h5")
        options = Logs.HDF5LogOptions(;
            filename,
            logging_policy = selected_policy,
        )
        history = run_logging_simulation(selected_policy; log_options = options)
        expected_times = collect(history["/"]["keep_continuous_state"].time)
        expected_data = collect(history["/"]["keep_continuous_state"].data)
        assert_selected_variables(history["/"])
        assert_selected_variables(history["/child"])
        Logs.close_log(history.log)

        # Reloading is the important regression check: stale metadata for a dropped variable
        # would make the loader search for a time-series group that was never created.
        loaded_log, loaded_history = Logs.load_hdf5_log(filename)
        assert_selected_variables(loaded_history)
        assert_selected_variables(loaded_history["child"])
        @test collect(loaded_history["keep_continuous_state"].time) == expected_times
        @test collect(loaded_history["keep_continuous_state"].data) == expected_data
        Logs.close_log(loaded_log)

        # Saving a filtered in-memory log follows a separate code path from direct HDF5
        # logging. It should nevertheless produce the same selected on-disk structure.
        basic_history = run_logging_simulation(selected_policy)
        saved_filename = joinpath(directory, "saved_filtered_log.h5")
        Logs.save_log_to_hdf5(saved_filename, basic_history.log)
        saved_log, saved_history = Logs.load_hdf5_log(saved_filename)
        assert_selected_variables(saved_history)
        assert_selected_variables(saved_history["child"])
        Logs.close_log(saved_log)

    end

end

@testset "regular state snapshots in HDF5 logs" begin

    # Direct-to-disk logging uses the same runtime functions but different TimeSeries
    # storage. Repeating the sparse-update regression here ensures snapshot fallback values
    # are appended correctly without relying on in-memory vector behavior.
    policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => Samplers.RegularSampler(1//2),
        ],
    )
    mktempdir() do directory

        filename = joinpath(directory, "regular_state_snapshots.h5")
        options = Logs.HDF5LogOptions(; filename, logging_policy = policy)
        history = run_sparse_update_simulation(policy; log_options = options)
        assert_regular_state_snapshots(history["/"])
        Logs.close_log(history.log)

        # Reloading verifies that both the snapshot values and their selected timestamps
        # were persisted, rather than only being visible through the live HDF5 vectors.
        loaded_log, loaded_history = Logs.load_hdf5_log(filename)
        assert_regular_state_snapshots(loaded_history)
        Logs.close_log(loaded_log)

    end

end

end # TestLoggingPolicies
