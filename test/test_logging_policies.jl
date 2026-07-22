module TestLoggingPolicies

using Test
using HDF5Vectors # Load the HDF5 extension.
using SystemsOfSystems
using SystemsOfSystems: LoggingPolicies, Logs, Samplers, Solvers

const SELECTED_VARIABLES = [
    :keep_constant,
    :keep_continuous_state,
    :keep_discrete_state,
    :keep_continuous_output,
    :keep_discrete_output,
]

function model_description(; include_child = true)

    models = if include_child
        (; child = model_description(; include_child = false))
    else
        (;)
    end

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

    models = if hasproperty(model, :child)
        (; child = rates(t, model.child))
    else
        (;)
    end

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

    models = if hasproperty(model, :child)
        (; child = updates(t, model.child))
    else
        (;)
    end

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

    if isnothing(log_options)
        log_options = Logs.BasicLogOptions(; logging_policy)
    end

    history, = simulate(
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

function assert_selected_variables(model_history)

    @test keys(model_history.constants) == (:keep_constant,)
    @test keys(model_history.continuous_states) == (:keep_continuous_state,)
    @test keys(model_history.discrete_states) == (:keep_discrete_state,)
    @test keys(model_history.continuous_outputs) == (:keep_continuous_output,)
    @test keys(model_history.discrete_outputs) == (:keep_discrete_output,)

end

@testset "variable sets" begin

    all_variables = LoggingPolicies.AllVariables()
    no_variables = LoggingPolicies.NoVariables()
    variable_list = LoggingPolicies.VariableList(["one", "two"])
    exclusion_list = LoggingPolicies.VariableExclusionList(; list = [:one, :two])

    @test LoggingPolicies.is_variable_in_set(:anything, all_variables)
    @test !LoggingPolicies.is_variable_in_set(:anything, no_variables)
    @test LoggingPolicies.is_variable_in_set(:one, variable_list)
    @test !LoggingPolicies.is_variable_in_set(:three, variable_list)
    @test !LoggingPolicies.is_variable_in_set(:one, exclusion_list)
    @test LoggingPolicies.is_variable_in_set(:three, exclusion_list)
    @test variable_list.list == [:one, :two]

end

@testset "sampling directives and built-in samplers" begin

    complete = Samplers.get_sampling_directive(0//1, Samplers.CompleteSampler())
    null = Samplers.get_sampling_directive(0//1, Samplers.NullSampler())
    states_only = Samplers.SamplingDirective(;
        log_states = true,
        log_outputs = false,
        log_models = false,
    )

    @test Samplers.should_log_states(complete)
    @test Samplers.should_log_outputs(complete)
    @test Samplers.should_log_models(complete)
    @test !Samplers.should_log_states(null)
    @test !Samplers.should_log_outputs(null)
    @test !Samplers.should_log_models(null)
    @test Samplers.get_sampling_directive(0//1, states_only) === states_only
    @test Samplers.should_log_states(states_only)
    @test !Samplers.should_log_outputs(states_only)
    @test !Samplers.should_log_models(states_only)

    # Positional and keyword construction both convert ordinary real values to exact time.
    positional = Samplers.RegularSampler(0.5, 0.25, true)
    keyword = Samplers.RegularSampler(;
        period = 1//2,
        offset = 1//4,
        continue_to_submodels = true,
    )
    @test positional.period == 1//2
    @test positional.offset == 1//4
    @test positional == keyword
    @test Samplers.RegularSampler(0.1).period == 1//10

    # The offset is the first sample. Off-grid behavior independently controls whether
    # descendant samplers are still consulted.
    before_offset = Samplers.get_sampling_directive(0//1, keyword)
    on_grid = Samplers.get_sampling_directive(3//4, keyword)
    off_grid = Samplers.get_sampling_directive(1//2, keyword)
    @test !Samplers.should_log_states(before_offset)
    @test Samplers.should_log_models(before_offset)
    @test Samplers.should_log_states(on_grid)
    @test Samplers.should_log_outputs(on_grid)
    @test Samplers.should_log_models(on_grid)
    @test !Samplers.should_log_states(off_grid)
    @test !Samplers.should_log_outputs(off_grid)
    @test Samplers.should_log_models(off_grid)

    # Invalid clocks are rejected during construction rather than failing in the loop.
    @test_throws ArgumentError Samplers.RegularSampler(0)
    @test_throws ArgumentError Samplers.RegularSampler(-1)
    @test_throws ArgumentError Samplers.RegularSampler(1//0)
    @test_throws ArgumentError Samplers.RegularSampler(1, 1//0)

end

@testset "model logging policies" begin

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

    root_sampler = Samplers.RegularSampler(1//2)
    child_policy = LoggingPolicies.ModelLoggingPolicy(;
        variable_set = LoggingPolicies.VariableList([:keep_constant]),
    )
    compact_rules = [
        r"^/$" => root_sampler,
        "/child\$" => child_policy,
    ]

    # Pair conversion supports both sampler shorthand and explicit model policies.
    root_rule = convert(LoggingPolicies.RegexLoggingPolicyRule, first(compact_rules))
    child_rule = LoggingPolicies.RegexLoggingPolicyRule(last(compact_rules))
    @test root_rule.expression == r"^/$"
    @test LoggingPolicies.get_sampler(root_rule.policy) === root_sampler
    @test child_rule.expression == r"/child$"
    @test child_rule.policy === child_policy

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

    defaulted = LoggingPolicies.RegexLoggingPolicy(compact_rules)
    @test LoggingPolicies.get_model_logging_policy(defaulted, "/other") isa
        LoggingPolicies.NullModelLoggingPolicy

    first_match = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/" => LoggingPolicies.NullModelLoggingPolicy(),
            r"^/$" => LoggingPolicies.AllPassModelLoggingPolicy(),
        ],
    )
    @test LoggingPolicies.get_model_logging_policy(first_match, "/") isa
        LoggingPolicies.NullModelLoggingPolicy

    all_pass_policy = LoggingPolicies.AllPassLoggingPolicy()
    uniform_policy = LoggingPolicies.UniformLoggingPolicy(; policy = child_policy)
    @test LoggingPolicies.get_model_logging_policy(all_pass_policy, "/anything") isa
        LoggingPolicies.AllPassModelLoggingPolicy
    @test LoggingPolicies.get_model_logging_policy(uniform_policy, "/anything") ===
        child_policy

end


@testset "variable selection in model histories" begin

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
    # allocating variable containers or recording samples.
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

@testset "regular sampling and subtree traversal" begin

    # An offset sampler rejects the initial discrete values and records both root and child
    # only at matching accepted times when the root gates its subtree.
    gated_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => Samplers.RegularSampler(1//2, 1//4),
            r"^/" => Samplers.CompleteSampler(),
        ],
    )
    gated_history = run_logging_simulation(gated_policy)
    for model_path in ("/", "/child")
        model_history = gated_history[model_path]
        @test model_history["keep_continuous_state"].time == [0.25, 0.75]
        @test model_history["keep_continuous_output"].time == [0.25, 0.75]
        @test model_history["keep_discrete_state"].time == [0.25, 0.75]
        @test model_history["keep_discrete_output"].time == [0.25, 0.75]
    end

    # Continuing traversal on rejected root samples lets the child retain its own complete
    # sampling behavior while the root remains decimated.
    traversing_policy = LoggingPolicies.RegexLoggingPolicy(
        [
            r"^/$" => Samplers.RegularSampler(1//2, 0, true),
            r"^/" => Samplers.CompleteSampler(),
        ],
    )
    traversing_history = run_logging_simulation(traversing_policy)
    @test traversing_history["/"]["keep_continuous_state"].time == [0.0, 0.5, 1.0]
    @test traversing_history["/"]["keep_discrete_state"].time == [0.0, 0.5, 1.0]
    @test traversing_history["/child"]["keep_continuous_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]
    @test traversing_history["/child"]["keep_discrete_state"].time ==
        [0.0, 0.25, 0.5, 0.75, 1.0]

end

@testset "independent state, output, and submodel directives" begin

    output_only = Samplers.SamplingDirective(;
        log_states = false,
        log_outputs = true,
        log_models = true,
    )
    policy = LoggingPolicies.RegexLoggingPolicy(
        [r"^/$" => output_only],
        LoggingPolicies.AllPassModelLoggingPolicy(),
    )
    history = run_logging_simulation(policy)

    @test isempty(history["/"]["keep_continuous_state"].time)
    @test isempty(history["/"]["keep_discrete_state"].time)
    @test !isempty(history["/"]["keep_continuous_output"].time)
    @test !isempty(history["/"]["keep_discrete_output"].time)
    @test !isempty(history["/child"]["keep_continuous_state"].time)
    @test !isempty(history["/child"]["keep_discrete_state"].time)

end

@testset "filtered HDF5 logs" begin

    selected_policy = LoggingPolicies.UniformLoggingPolicy(;
        policy = LoggingPolicies.ModelLoggingPolicy(;
            variable_set = LoggingPolicies.VariableList(SELECTED_VARIABLES),
        ),
    )

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

        loaded_log, loaded_history = Logs.load_hdf5_log(filename)
        assert_selected_variables(loaded_history)
        assert_selected_variables(loaded_history["child"])
        @test collect(loaded_history["keep_continuous_state"].time) == expected_times
        @test collect(loaded_history["keep_continuous_state"].data) == expected_data
        Logs.close_log(loaded_log)

        # Saving a filtered in-memory log uses the same on-disk representation and retains
        # the selected structure when loaded again.
        basic_history = run_logging_simulation(selected_policy)
        saved_filename = joinpath(directory, "saved_filtered_log.h5")
        Logs.save_log_to_hdf5(saved_filename, basic_history.log)
        saved_log, saved_history = Logs.load_hdf5_log(saved_filename)
        assert_selected_variables(saved_history)
        assert_selected_variables(saved_history["child"])
        Logs.close_log(saved_log)

    end

end


end # TestLoggingPolicies
