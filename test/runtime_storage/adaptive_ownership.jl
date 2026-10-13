module StateAdaptiveOwnershipTests

using Test, SystemsOfSystems

const SoS = SystemsOfSystems
const CP = SoS.ContinuousProblems
const Solvers = SoS.Solvers
const HAS_WORKSPACE = isdefined(CP, :create_provisional_workspace)

mutable struct RetainingErrorPolicy
    calls::Int
    states::Vector{Any}
    snapshots::Vector{Any}
    events::Vector{Symbol}
    throw_next::Bool
end

RetainingErrorPolicy() = RetainingErrorPolicy(0, Any[], Any[], Symbol[], false)

# Error policies can retain and mutate their persistent full and embedded descriptors.
# Their effects must occur after all rates callbacks, and survive later rejected attempts.
function CP.normalized_error(
    policy::RetainingErrorPolicy,
    description,
    state::SoS.ModelStateDescription,
    embedded_state::SoS.ModelStateDescription,
    absolute_tolerance,
    relative_tolerance,
)

    policy.calls += 1
    push!(policy.events, :error)
    state.continuous_states.x[1] += 0.01
    embedded_state.continuous_states.x[1] -= 0.02
    push!(policy.states, (state, embedded_state))
    push!(policy.snapshots, (
        copy(state.continuous_states.x), copy(embedded_state.continuous_states.x),
    ))
    if policy.throw_next
        policy.throw_next = false
        error("normalized error policy failure")
    end
    return policy.calls == 1 ? 32. : 0.25

end

function initialization()

    description = ModelDescription(;
        continuous_states = (; x = [1.], held = [9.]),
        continuous_outputs = (; sample = 0),
    )
    return SoS.create_initialization_artifacts(description, SoS.initialization_context())

end

function adaptive_integrator(problem, state, borrowed)

    if borrowed
        integrator = Solvers.create_integrator(Solvers.DormandPrince54Options(;
            initial_dt = 1//2, max_dt = 1//2, abs_tol = 1., rel_tol = 0.,
        ), problem, state)
        if HAS_WORKSPACE
            @test integrator.workspace !== nothing
        end
        return integrator
    end
    controller = Solvers.EmbeddedAdaptiveController(0.5, 0.5, 1., 0., 0.8, 20)
    return Solvers.RungeKuttaIntegrator(Solvers.DORMAND_PRINCE_54_TABLEAU, controller)

end

function assert_retained(policy, models, snapshots)

    @test all(model.x == saved for (model, saved) in zip(models, snapshots))
    for ((state, embedded), (state_copy, embedded_copy)) in
        zip(policy.states, policy.snapshots)
        @test state.continuous_states.x == state_copy
        @test embedded.continuous_states.x == embedded_copy
        @test state !== embedded
        @test state.continuous_states.x !== embedded.continuous_states.x
    end

end

function adaptive_case(borrowed)

    initialized = initialization()
    policy = RetainingErrorPolicy()
    models = Any[]
    snapshots = Vector{Float64}[]
    times = Float64[]
    rates = (t, model) -> begin
        push!(policy.events, :rates)
        push!(models, model)
        push!(snapshots, copy(model.x))
        push!(times, t)
        return RatesOutput(;
            rates = (; x = copy(model.x)),
            outputs = (; sample = length(models)),
            stop = length(models) == 1,
        )
    end
    problem = CP.ContinuousProblem(initialized.ommd, rates, policy)
    integrator = adaptive_integrator(problem, initialized.msd, borrowed)
    first = Solvers.step!(integrator, problem,
        Solvers.StepRequest(0//1, 1//2, initialized.msd, SoS.NO_T_NEXT))

    @test first isa Solvers.AcceptedStep
    @test policy.calls == 2
    @test length(models) == 14
    @test first.t_end < 1//2
    @test first.rates_at_start.outputs.sample == 8
    @test !first.rates_at_start.stop
    @test first.state_at_start === initialized.msd
    @test first.state_at_end === policy.states[2][1]
    @test initialized.msd.continuous_states.x == [1.]
    @test policy.events == vcat(fill(:rates, 7), [:error], fill(:rates, 7), [:error])
    @test all(model.held === initialized.msd.continuous_states.held for model in models)
    assert_retained(policy, models, snapshots)
    first_snapshot = copy(first.state_at_end.continuous_states.x)

    # The next invocation replaces provisional values again. Retained rejected and
    # embedded states, accepted states, and every earlier callback array must survive.
    second = Solvers.step!(integrator, problem,
        Solvers.StepRequest(first.t_end, 1//1, first.state_at_end, SoS.NO_T_NEXT))
    @test second isa Solvers.AcceptedStep
    @test policy.calls == 3
    @test length(models) == 21
    @test second.state_at_start === first.state_at_end
    @test first.state_at_end.continuous_states.x == first_snapshot
    @test second.state_at_end !== first.state_at_end
    assert_retained(policy, models, snapshots)
    if borrowed && HAS_WORKSPACE
        @test !integrator.workspace.active
        @test all(state !== integrator.workspace.state && embedded !== integrator.workspace.state
            for (state, embedded) in policy.states)
    end
    return (; policy, models, snapshots, times, first, second)

end

@testset "Adaptive callback models and error-policy states are persistent" begin

    persistent = adaptive_case(false)
    selected = adaptive_case(true)
    @test selected.snapshots == persistent.snapshots
    @test selected.times == persistent.times
    @test selected.policy.events == persistent.policy.events
    @test selected.policy.snapshots == persistent.policy.snapshots
    @test selected.first.t_end == persistent.first.t_end
    @test selected.second.t_end == persistent.second.t_end
    @test selected.first.state_at_end.continuous_states.x ==
        persistent.first.state_at_end.continuous_states.x
    @test selected.second.state_at_end.continuous_states.x ==
        persistent.second.state_at_end.continuous_states.x

end

@testset "Error-policy exceptions leave the workspace available" begin

    initialized = initialization()
    policy = RetainingErrorPolicy()
    policy.throw_next = true
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = copy(model.x))), policy)
    integrator = adaptive_integrator(problem, initialized.msd, true)
    request = Solvers.StepRequest(0//1, 1//2, initialized.msd, SoS.NO_T_NEXT)
    @test_throws ErrorException Solvers.step!(integrator, problem, request)
    @test initialized.msd.continuous_states.x == [1.]
    @test integrator.controller.next_dt == 0.5
    if HAS_WORKSPACE
        @test !integrator.workspace.active
    end
    accepted = Solvers.step!(integrator, problem, request)
    @test accepted isa Solvers.AcceptedStep
    @test policy.calls == 2
    assert_retained(policy, Any[], Vector{Float64}[])

end

@testset "Independent integrators do not share provisional shells" begin

    initialized = initialization()
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = copy(model.x))))
    first = adaptive_integrator(problem, initialized.msd, true)
    second = adaptive_integrator(problem, initialized.msd, true)
    if HAS_WORKSPACE
        @test first.workspace !== second.workspace
        @test first.workspace.state !== second.workspace.state
        @test first.workspace.state.continuous_states.x ===
            second.workspace.state.continuous_states.x
    end
    request = Solvers.StepRequest(0//1, 1//2, initialized.msd, SoS.NO_T_NEXT)
    first_result = Solvers.step!(first, problem, request)
    first_snapshot = copy(first_result.state_at_end.continuous_states.x)
    second_result = Solvers.step!(second, problem, request)
    @test first_result.t_end == second_result.t_end
    @test first_result.state_at_end.continuous_states.x == first_snapshot
    @test first_snapshot == second_result.state_at_end.continuous_states.x
    @test first_result.state_at_end !== second_result.state_at_end

end

end # StateAdaptiveOwnershipTests
