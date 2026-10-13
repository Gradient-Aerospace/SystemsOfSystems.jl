module StageExtensionTests

using Test, SystemsOfSystems

const SoS = SystemsOfSystems
const CP = SoS.ContinuousProblems
const Solvers = SoS.Solvers
const MODE = only(ARGS)
@assert MODE in ("weighted", "variable")
const EVENTS = Symbol[]

struct Value
    x::Float64
end
Base.:*(gain::Number, rate::Value) = Value(gain * rate.x)
Base.:+(left::Value, right::Value) = Value(left.x + right.x)

@testset "Late arithmetic helper extension preserves dispatch: $MODE" begin

    initialized = SoS.create_initialization_artifacts(
        ModelDescription(; continuous_states = (; x = Value(0.))),
        SoS.initialization_context(),
    )
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = Value(1.))))
    solver = Solvers.create_integrator(Solvers.RungeKutta4Options(; dt = 1//4),
        problem, initialized.msd)
    @test solver.workspace !== nothing
    @test CP.provisional_workspace_available(solver.workspace, problem, initialized.msd)

    # Each mode runs in a fresh process, so it independently proves the corresponding
    # helper extension invalidates a previously available private stage capability.
    if MODE == "weighted"
        @eval function CP.weighted_rate(gains::Tuple{G}, rates::Tuple{Value}) where {G}
            push!(EVENTS, :weighted)
            return first(gains) * first(rates)
        end
    else
        @eval function CP.propagate_variable(x::Value, gains::Tuple, rates::Tuple)
            push!(EVENTS, :variable)
            return (x + CP.weighted_rate(gains, rates))::Value
        end
    end
    @test !CP.provisional_workspace_available(solver.workspace, problem, initialized.msd)
    request = Solvers.StepRequest(0//1, 1//4, initialized.msd, SoS.NO_T_NEXT)
    actual = Base.invokelatest(Solvers.step!, solver, problem, request)
    observed = copy(EVENTS)
    empty!(EVENTS)
    oracle = Solvers.RungeKuttaIntegrator(
        Solvers.RUNGE_KUTTA_4_TABLEAU, Solvers.FixedStepController(0.25))
    expected = Base.invokelatest(Solvers.step!, oracle, problem, request)
    @test !isempty(observed)
    @test EVENTS == observed
    @test actual.state_at_end.continuous_states.x == expected.state_at_end.continuous_states.x
    @test !solver.workspace.active

end

end # StageExtensionTests
