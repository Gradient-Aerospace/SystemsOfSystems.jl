module StageTransportOwnershipTests

using Test, SystemsOfSystems

const SoS = SystemsOfSystems
const CP = SoS.ContinuousProblems
const Solvers = SoS.Solvers
@assert isdefined(CP, :accessed_provisional_state)

function initialize(description)
    return SoS.create_initialization_artifacts(description, SoS.initialization_context())
end

function integrator(problem, state; persistent = false, method = Solvers.RUNGE_KUTTA_4_TABLEAU,
    controller = Solvers.FixedStepController(0.25))
    if persistent
        return Solvers.RungeKuttaIntegrator(method, controller)
    end
    workspace = CP.create_provisional_workspace(problem, state)
    @test workspace !== nothing
    return Solvers.RungeKuttaIntegrator(method, controller, workspace)
end

function assert_empty(integrator)
    workspace = integrator.workspace
    workspace === nothing && return
    @test !workspace.active
    store = workspace.stage_storage
    @test store === nothing || isempty(store.stages)
end

const EVENTS = Symbol[]
struct OrderedState
    value::Float64
    label::Symbol
end
function Base.:*(gain::Number, rate::OrderedState)
    push!(EVENTS, rate.label)
    return OrderedState(gain * rate.value, rate.label)
end
function Base.:+(left::OrderedState, right::OrderedState)
    push!(EVENTS, :add)
    return OrderedState(left.value + right.value, left.label)
end

@testset "Literal stage access preserves exact operation and subtree order" begin

    source = SoS.ModelStateDescription{Nothing}(;
        continuous_states = (; x = OrderedState(1., :parent)),
        models = (;
            child = SoS.ModelStateDescription{Nothing}(;
                continuous_states = (; x = OrderedState(2., :child))),
            held = SoS.ModelStateDescription{Nothing}(;
                continuous_states = (; x = [9.])),
        ),
    )
    initialized = initialize(ModelDescription(; continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> RatesOutput())
    workspace = CP.create_provisional_workspace(problem, source)
    @test workspace !== nothing
    stages = ntuple(3) do index
        RatesOutput(; rates = (; x = OrderedState(index, Symbol(:parent, index))),
            models = (; child = RatesOutput(;
                rates = (; x = OrderedState(index, Symbol(:child, index))))))
    end
    gains = (0.25, 0., 0.125)
    empty!(EVENTS)
    expected = CP.propagate(source, gains, stages)
    expected_events = copy(EVENTS)
    @test :parent2 in expected_events
    @test :child2 in expected_events
    for storage in (stages, collect(stages))
        empty!(EVENTS)
        actual = CP.accessed_persistent_state(source, gains, storage)
        @test EVENTS == expected_events
        @test actual.continuous_states.x == expected.continuous_states.x
        @test actual.models.child.continuous_states.x == expected.models.child.continuous_states.x
        @test actual.models.held === source.models.held
        empty!(EVENTS)
        borrowed = CP.accessed_provisional_state(workspace, source, gains, storage)
        @test EVENTS == expected_events
        @test borrowed.continuous_states.x == expected.continuous_states.x
        @test borrowed.models.child.continuous_states.x == expected.models.child.continuous_states.x
        @test borrowed.models.held !== source.models.held
        @test borrowed.models.held.continuous_states.x === source.models.held.continuous_states.x
    end

end

function changing_stage_case(storage; persistent = false, missing_later = false)

    initialized = initialize(ModelDescription(;
        continuous_states = (; x = [1.], held = [7.]),
        continuous_outputs = (; call = 0),
    ))
    attempt_index = Ref(1)
    call_index = Ref(0)
    models = Any[]
    snapshots = Any[]
    times = Float64[]
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> begin
        call_index[] += 1
        push!(models, model)
        push!(snapshots, copy(model.x))
        push!(times, t)
        rates = if missing_later && call_index[] == 2
            (;)
        elseif attempt_index[] == 1 && call_index[] == 1
            (; x = copy(model.x))
        else
            (; x = copy(model.x), held = [2.])
        end
        return RatesOutput(; rates, outputs = (; call = call_index[]))
    end)
    solver = integrator(problem, initialized.msd; persistent)
    results = Any[]
    state = initialized.msd
    if missing_later
        error = try
            Solvers.attempt_solutions(solver, problem, 0//1, 1//4, 0.25, state, storage)
            nothing
        catch exception
            exception
        end
        @test error !== nothing
        @test call_index[] == 2
        assert_empty(solver)
        return (; error_type = typeof(error), calls = call_index[], times, snapshots)
    end
    for attempt in 1:3
        attempt_index[] = attempt
        call_index[] = 0
        result = Solvers.attempt_solutions(solver, problem, (attempt - 1)//4,
            attempt//4, 0.25, state, storage)
        push!(results, result)
        state = result.state_at_end
        @test call_index[] == 4
        assert_empty(solver)
    end
    @test all(model.x == snapshot for (model, snapshot) in zip(models, snapshots))
    @test results[1].state_at_end.continuous_states.held === initialized.msd.continuous_states.held
    @test results[2].state_at_end.continuous_states.held !== initialized.msd.continuous_states.held
    return (; results, snapshots, times)

end

@testset "Heterogeneous stages and later attempts use captured values once" begin

    reference = changing_stage_case(Val(:vector); persistent = true)
    for storage in (Val(:tuple), Val(:vector))
        candidate = changing_stage_case(storage)
        @test candidate.snapshots == reference.snapshots
        @test candidate.times == reference.times
        for (actual, expected) in zip(candidate.results, reference.results)
            @test actual.state_at_end.continuous_states == expected.state_at_end.continuous_states
            @test actual.rates_at_start.outputs == expected.rates_at_start.outputs
        end
        actual_error = changing_stage_case(storage; missing_later = true)
        expected_error = changing_stage_case(storage; persistent = true, missing_later = true)
        @test actual_error == expected_error
    end

end

const REENTRY_ACTION = Ref{Any}(nothing)
const REENTRY_GAIN = Ref(0.)
const REENTRY_PENDING = Ref(false)
struct ReentrantState
    value::Float64
end
function Base.:*(gain::Number, rate::ReentrantState)
    if REENTRY_PENDING[] && gain == REENTRY_GAIN[]
        REENTRY_PENDING[] = false
        REENTRY_ACTION[]()
    end
    return ReentrantState(gain * rate.value)
end
Base.:+(left::ReentrantState, right::ReentrantState) = ReentrantState(left.value + right.value)

struct AlwaysAccept end
CP.normalized_error(::AlwaysAccept, description, state::SoS.ModelStateDescription,
    embedded::SoS.ModelStateDescription, absolute_tolerance, relative_tolerance) = 0.25

function arithmetic_reentry_case(method; persistent = false, embedded = false)

    initialized = initialize(ModelDescription(;
        continuous_states = (; x = ReentrantState(0.))))
    nested_state = SoS.copy_model_state_description_except(initialized.msd;
        continuous_states = (; x = ReentrantState(100.)))
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = ReentrantState(1.))), AlwaysAccept())
    controller = embedded ? Solvers.EmbeddedAdaptiveController(0.25, 0.25, 1., 0., 0.8, 20) :
        Solvers.FixedStepController(0.25)
    solver = integrator(problem, initialized.msd; persistent, method, controller)
    nested = Ref{Any}(nothing)
    triggered = Ref(false)
    REENTRY_ACTION[] = () -> begin
        triggered[] = true
        if !persistent
            @test solver.workspace.active
            @test !isempty(solver.workspace.stage_storage.stages)
        end
        nested[] = Solvers.step!(solver, problem,
            Solvers.StepRequest(0//1, 1//4, nested_state, SoS.NO_T_NEXT))
    end
    REENTRY_GAIN[] = 0.25 * first(embedded ? method.embedded_b : method.b)
    REENTRY_PENDING[] = true
    result = Solvers.step!(solver, problem,
        Solvers.StepRequest(0//1, 1//4, initialized.msd, SoS.NO_T_NEXT))
    @test triggered[]
    assert_empty(solver)
    return (; outer = result.state_at_end.continuous_states.x.value,
        inner = nested[].state_at_end.continuous_states.x.value)

end

@testset "Custom arithmetic cannot reenter an active stage bank" begin

    # The RK4 final first weight differs from every provisional coefficient. This invokes
    # a custom arithmetic operation while the bank is live, without replacing an adapter.
    method = Solvers.RUNGE_KUTTA_4_TABLEAU
    @test arithmetic_reentry_case(method) == arithmetic_reentry_case(method; persistent = true)

    # This embedded method has a first embedded coefficient distinct from provisional
    # and primary weights, so the second check reaches the still-live bank's last consumer.
    embedded_method = Solvers.ExplicitRungeKuttaTableau(((0.5,),), (0.25, 0.75), (0.5,);
        embedded_b = (0.625, 0.375), order = 2, embedded_order = 1)
    @test arithmetic_reentry_case(embedded_method; embedded = true) ==
        arithmetic_reentry_case(embedded_method; persistent = true, embedded = true)

end

@testset "Public stage results remain persistent across private attempts" begin

    initialized = initialize(ModelDescription(; continuous_states = (; x = [1.])))
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = copy(model.x))))
    solver = integrator(problem, initialized.msd)
    public_attempt = Solvers.evaluate_stages(solver.method, problem,
        0//1, 1//4, 0.25, initialized.msd, solver.workspace)
    @test public_attempt.stages isa Tuple
    retained = [copy(stage.rates.x) for stage in public_attempt.stages]
    for _ in 1:2
        Solvers.step!(solver, problem,
            Solvers.StepRequest(0//1, 1//4, initialized.msd, SoS.NO_T_NEXT))
    end
    @test [stage.rates.x for stage in public_attempt.stages] == retained
    assert_empty(solver)

end

# A failed callback or error policy must release both the descriptor lease and every stage
# payload reference before the same integrator can be used for a later attempt.
struct ThrowingPolicy
    workspace::Base.RefValue{Any}
end
function CP.normalized_error(policy::ThrowingPolicy, description,
    state::SoS.ModelStateDescription, embedded::SoS.ModelStateDescription,
    absolute_tolerance, relative_tolerance)

    @test policy.workspace[].active
    @test !isempty(policy.workspace[].stage_storage.stages)
    error("requested error-policy failure")

end

@testset "Exceptions clear the bank before releasing its lease" begin

    initialized = initialize(ModelDescription(; continuous_states = (; x = [1.])))
    call_index = Ref(0)
    should_throw = Ref(true)
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> begin
        call_index[] += 1
        should_throw[] && call_index[] == 2 && error("requested rates failure")
        return RatesOutput(; rates = (; x = copy(model.x)))
    end)
    solver = integrator(problem, initialized.msd)
    request = Solvers.StepRequest(0//1, 1//4, initialized.msd, SoS.NO_T_NEXT)
    @test_throws ErrorException Solvers.step!(solver, problem, request)
    @test call_index[] == 2
    assert_empty(solver)
    should_throw[] = false
    @test Solvers.step!(solver, problem, request) isa Solvers.AcceptedStep
    assert_empty(solver)

    workspace_ref = Ref{Any}(nothing)
    adaptive_problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = copy(model.x))),
        ThrowingPolicy(workspace_ref))
    adaptive = Solvers.create_integrator(Solvers.DormandPrince54Options(;
        initial_dt = 1//4, max_dt = 1//4), adaptive_problem, initialized.msd)
    @test adaptive.workspace !== nothing
    workspace_ref[] = adaptive.workspace
    @test_throws ErrorException Solvers.step!(adaptive, adaptive_problem, request)
    assert_empty(adaptive)

end

# A RatesOutput property extension is an observable propagation protocol. Keep this last:
# the conservative capability guard then disables stage access throughout this process.
const PROPERTY_READS = Symbol[]

@testset "Late RatesOutput properties select persistent fallback" begin

    initialized = initialize(ModelDescription(; continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = 1.)))
    solver = integrator(problem, initialized.msd)
    @test CP.provisional_workspace_available(solver.workspace, problem, initialized.msd)
    @eval function Base.getproperty(output::SoS.RatesOutput, name::Symbol)
        push!(PROPERTY_READS, name)
        return getfield(output, name)
    end
    @test !CP.provisional_workspace_available(solver.workspace, problem, initialized.msd)
    request = Solvers.StepRequest(0//1, 1//4, initialized.msd, SoS.NO_T_NEXT)
    empty!(PROPERTY_READS)
    actual = Base.invokelatest(Solvers.step!, solver, problem, request)
    observed = copy(PROPERTY_READS)
    empty!(PROPERTY_READS)
    oracle = integrator(problem, initialized.msd; persistent = true)
    expected = Base.invokelatest(Solvers.step!, oracle, problem, request)
    @test PROPERTY_READS == observed
    @test !isempty(observed)
    @test actual.state_at_end.continuous_states == expected.state_at_end.continuous_states
    assert_empty(solver)

end

end # StageTransportOwnershipTests
