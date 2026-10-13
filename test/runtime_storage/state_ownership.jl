module StateWorkspaceTests

using Test
using SystemsOfSystems

const SoS = SystemsOfSystems
const CP = SoS.ContinuousProblems
const Solvers = SoS.Solvers
const HAS_WORKSPACE = isdefined(CP, :create_provisional_workspace)
const OPERATOR_EVENTS = Tuple{Symbol, Symbol}[]

# Custom arithmetic makes operator order observable, independently of floating-point values.
struct OrderedValue
    label::Symbol
    value::Float64
end
function Base.:*(gain::Number, value::OrderedValue)
    push!(OPERATOR_EVENTS, (:multiply, value.label))
    return OrderedValue(value.label, gain * value.value)
end
function Base.:+(left::OrderedValue, right::OrderedValue)
    push!(OPERATOR_EVENTS, (:add, left.label))
    return OrderedValue(left.label, left.value + right.value)
end

function artifacts(description)
    return SoS.create_initialization_artifacts(description, SoS.initialization_context())
end

function persistent_integrator()
    return Solvers.RungeKuttaIntegrator(
        Solvers.RUNGE_KUTTA_4_TABLEAU, Solvers.FixedStepController(0.25),
    )
end

function selected_integrator(problem, state)
    return Solvers.create_integrator(Solvers.RungeKutta4Options(; dt = 1//4), problem, state)
end

function step_request(state, start = 0//1)
    return Solvers.StepRequest(start, start + 1//4, state, SoS.NO_T_NEXT)
end

@testset "provisional workspace owns shells but preserves payload ownership" begin

    base = SoS.ModelStateDescription{Nothing}(;
        constants = (; table = [7.]),
        continuous_states = (; x = OrderedValue(:parent, 1.)),
        models = (;
            moving = SoS.ModelStateDescription{Nothing}(;
                continuous_states = (; x = [1., 2.], held = [9.]),
                discrete_states = (; gain = 3.),
            ),
            held = SoS.ModelStateDescription{Nothing}(;
                continuous_states = (; x = [4.]),
            ),
        ),
    )
    description = artifacts(ModelDescription(; continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(description.ommd, (t, model) -> RatesOutput())
    stages = (
        RatesOutput(; rates = (; x = OrderedValue(:first, 2.)),
            models = (; moving = RatesOutput(; rates = (; x = [2., 4.])))),
        RatesOutput(; rates = (; x = OrderedValue(:second, 3.)),
            models = (; moving = RatesOutput(; rates = (; x = [3., 6.])))),
    )
    gains = (0.25, 0.)
    empty!(OPERATOR_EVENTS)
    reference = CP.propagate(base, gains, stages)
    expected_events = copy(OPERATOR_EVENTS)
    @test !isempty(expected_events)
    @test (:multiply, :second) in expected_events

    if HAS_WORKSPACE

        workspace = CP.create_provisional_workspace(problem, base)
        @test workspace !== nothing
        owned_moving = workspace.state.models.moving
        owned_held = workspace.state.models.held
        @test workspace.state !== base
        @test owned_moving !== base.models.moving
        @test owned_held !== base.models.held
        @test workspace.state.static === base.static

        empty!(OPERATOR_EVENTS)
        provisional = CP.provisional_state(workspace, base, gains, stages)
        @test OPERATOR_EVENTS == expected_events
        @test provisional.continuous_states.x == reference.continuous_states.x
        @test provisional.models.moving.continuous_states.x == [1.5, 3.]
        @test provisional.models.moving.continuous_states.held ===
            base.models.moving.continuous_states.held
        @test provisional.models.held.continuous_states.x === base.models.held.continuous_states.x
        retained = SoS.model(provisional)
        retained_x = copy(retained.moving.x)

        # An omitted branch must not replace the workspace's owned link with a base node.
        CP.provisional_state(workspace, base, (1.,), (RatesOutput(),))
        @test workspace.state.models.moving === owned_moving
        @test workspace.state.models.held === owned_held
        CP.provisional_state(workspace, base, (0.5, 0.), stages)
        @test retained.moving.x == retained_x
        @test base.models.moving.continuous_states.x == [1., 2.]
        @test workspace.state.models.moving.continuous_states.x !== retained.moving.x

        # The next step can replace noncontinuous fields without changing descriptor types.
        child = SoS.copy_model_state_description_except(base.models.moving;
            discrete_states = (; gain = 8.))
        replacement = SoS.copy_model_state_description_except(base;
            constants = (; table = [11.]), models = (; moving = child, held = base.models.held))
        CP.provisional_state(workspace, replacement, gains, stages)
        @test workspace.state.static === replacement.static
        @test workspace.state.models.moving.discrete_states.gain == 8.
        @test retained.table === base.constants.table
        @test base.constants.table == [7.]

    end

end

function run_retained_arrays(use_workspace)

    initialized = artifacts(ModelDescription(; continuous_states = (; x = [1.], held = [7.])))
    models = Any[]
    snapshots = Vector{Float64}[]
    shared_derivative = [0.]
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> begin
        push!(models, model)
        push!(snapshots, copy(model.x))
        shared_derivative[1] = model.x[1]
        return RatesOutput(; rates = (; x = shared_derivative))
    end)
    integrator = if use_workspace
        selected_integrator(problem, initialized.msd)
    else
        persistent_integrator()
    end
    state = initialized.msd
    results = Any[]
    for index in 0:3
        result = Solvers.step!(integrator, problem, step_request(state, index//4))
        push!(results, result)
        state = result.state_at_end
    end
    @test all(model.x == saved for (model, saved) in zip(models, snapshots))
    @test all(model.held === initialized.msd.continuous_states.held for model in models)
    return (; models, snapshots, results, state)

end

@testset "retained callback arrays and public accepted results remain persistent" begin

    reference = run_retained_arrays(false)
    selected = run_retained_arrays(true)
    @test selected.snapshots == reference.snapshots
    @test selected.state.continuous_states.x == reference.state.continuous_states.x
    @test length(selected.models) == 16
    for (left, right) in zip(selected.results, reference.results)
        @test left.state_at_start.continuous_states.x == right.state_at_start.continuous_states.x
        @test left.state_at_end.continuous_states.x == right.state_at_end.continuous_states.x
    end
    @test selected.results[1].state_at_end !== selected.results[end].state_at_end

end

@testset "reentrant use and exceptions release the provisional lease" begin

    initialized = artifacts(ModelDescription(; continuous_states = (; x = [0.])))
    nested_state = SoS.copy_model_state_description_except(initialized.msd;
        continuous_states = (; x = [100.]))
    reference_problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = [1.])))
    reference = Solvers.step!(persistent_integrator(), reference_problem,
        step_request(initialized.msd))
    nested_reference = Solvers.step!(persistent_integrator(), reference_problem,
        step_request(nested_state))
    current_integrator = Ref{Any}(nothing)
    current_problem = Ref{Any}(nothing)
    nested_result = Ref{Any}(nothing)
    entered = Ref(false)
    should_throw = Ref(false)
    calls = Ref(0)
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> begin
        calls[] += 1
        if !entered[]
            entered[] = true
            nested_result[] = Solvers.step!(current_integrator[], current_problem[],
                step_request(nested_state))
        end
        should_throw[] && t > 0. && error("Intentional provisional callback failure.")
        return RatesOutput(; rates = (; x = [1.]))
    end)
    integrator = selected_integrator(problem, initialized.msd)
    current_integrator[] = integrator
    current_problem[] = problem
    @test calls[] == 0
    result = Solvers.step!(integrator, problem, step_request(initialized.msd))
    @test result.state_at_end.continuous_states.x == reference.state_at_end.continuous_states.x
    @test nested_result[].state_at_end.continuous_states.x ==
        nested_reference.state_at_end.continuous_states.x
    @test calls[] == 8
    if HAS_WORKSPACE
        @test !integrator.workspace.active
    end

    should_throw[] = true
    @test_throws ErrorException Solvers.step!(integrator, problem, step_request(initialized.msd))
    if HAS_WORKSPACE
        @test !integrator.workspace.active
    end
    should_throw[] = false
    later = Solvers.step!(integrator, problem, step_request(initialized.msd))
    @test later.state_at_end.continuous_states.x == reference.state_at_end.continuous_states.x
    @test nested_result[].state_at_end.continuous_states.x ==
        nested_reference.state_at_end.continuous_states.x

end

const RETAINED_DESCRIPTORS = Any[]
struct DescriptorRetainer
    x::Float64
end
DescriptorRetainer(; x) = DescriptorRetainer(x)
function SoS.model(description::SoS.ModelStateDescription{DescriptorRetainer})
    push!(RETAINED_DESCRIPTORS, description)
    return DescriptorRetainer(description.continuous_states.x)
end

struct RatesRetainer end
(::RatesRetainer)(t, model) = RatesOutput(; rates = (; x = 1.))
function CP.evaluate_rates(
    problem::CP.ContinuousProblem{TD, RatesRetainer, EP},
    t::AbstractFloat,
    state::SoS.ModelStateDescription,
) where {TD, EP}
    push!(RETAINED_DESCRIPTORS, state)
    output = problem.rates_fcn(t, SoS.model(state))
    SoS.validate_rates_output(problem.typed_description, output)
    return output
end

@testset "descriptor-retaining model and rates adapters use persistent fallback" begin

    for (model_type, rates_function) in (
        (DescriptorRetainer, (t, model) -> RatesOutput(; rates = (; x = 1.))),
        (Nothing, RatesRetainer()),
    )
        initialized = artifacts(ModelDescription(; type = model_type,
            continuous_states = (; x = 0.)))
        problem = CP.ContinuousProblem(initialized.ommd, rates_function)
        empty!(RETAINED_DESCRIPTORS)
        integrator = selected_integrator(problem, initialized.msd)
        @test isempty(RETAINED_DESCRIPTORS)
        if HAS_WORKSPACE
            @test integrator.workspace === nothing
        end
        Solvers.step!(integrator, problem, step_request(initialized.msd))
        saved_values = [state.continuous_states.x for state in RETAINED_DESCRIPTORS]
        retained = copy(RETAINED_DESCRIPTORS)
        Solvers.step!(integrator, problem, step_request(initialized.msd))
        @test [state.continuous_states.x for state in retained] == saved_values
        @test length(retained) == 4
    end

end

struct LateRetainer
    x::Float64
end
LateRetainer(; x) = LateRetainer(x)

@testset "later adapter methods invalidate provisional capability" begin

    initialized = artifacts(ModelDescription(; type = LateRetainer,
        continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = 1.)))
    integrator = selected_integrator(problem, initialized.msd)
    if HAS_WORKSPACE
        @test integrator.workspace !== nothing
    end
    empty!(RETAINED_DESCRIPTORS)

    # Enter the new Julia method world explicitly so this tests a method added between uses.
    @eval function SoS.model(description::SoS.ModelStateDescription{LateRetainer})
        push!(RETAINED_DESCRIPTORS, description)
        return LateRetainer(description.continuous_states.x)
    end
    Base.invokelatest(Solvers.step!, integrator, problem, step_request(initialized.msd))
    @test length(RETAINED_DESCRIPTORS) == 4
    @test [state.continuous_states.x for state in RETAINED_DESCRIPTORS] == [0., 0.125, 0.125, 0.25]
    if HAS_WORKSPACE
        @test !integrator.workspace.compatible
        @test !integrator.workspace.active
    end

end

struct LateRatesRetainer end
(::LateRatesRetainer)(t, model) = RatesOutput(; rates = (; x = 1.))

@testset "later rates adapters and a different problem keep persistent descriptors" begin

    initialized = artifacts(ModelDescription(; continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(initialized.ommd, LateRatesRetainer())
    integrator = selected_integrator(problem, initialized.msd)
    if HAS_WORKSPACE
        @test integrator.workspace !== nothing
    end
    empty!(RETAINED_DESCRIPTORS)
    @eval function CP.evaluate_rates(
        problem::CP.ContinuousProblem{TD, LateRatesRetainer, EP},
        t::AbstractFloat,
        state::SoS.ModelStateDescription,
    ) where {TD, EP}
        push!(RETAINED_DESCRIPTORS, state)
        output = problem.rates_fcn(t, SoS.model(state))
        SoS.validate_rates_output(problem.typed_description, output)
        return output
    end
    Base.invokelatest(Solvers.step!, integrator, problem, step_request(initialized.msd))
    @test [state.continuous_states.x for state in RETAINED_DESCRIPTORS] ==
        [0., 0.125, 0.125, 0.25]
    if HAS_WORKSPACE
        @test !integrator.workspace.compatible
    end

    standard_problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = 1.)))
    other_integrator = selected_integrator(standard_problem, initialized.msd)
    empty!(RETAINED_DESCRIPTORS)
    Solvers.step!(other_integrator, CP.ContinuousProblem(initialized.ommd, RatesRetainer()),
        step_request(initialized.msd))
    @test [state.continuous_states.x for state in RETAINED_DESCRIPTORS] ==
        [0., 0.125, 0.125, 0.25]

end

const CONSTRUCTOR_EVENTS = Symbol[]
struct ConstructedLeaf
    x::Float64
end
function ConstructedLeaf(; x)
    push!(CONSTRUCTOR_EVENTS, :leaf)
    return ConstructedLeaf(x)
end
struct ConstructedParent
    a::ConstructedLeaf
    b::ConstructedLeaf
end
function ConstructedParent(; a, b)
    push!(CONSTRUCTOR_EVENTS, :parent)
    return ConstructedParent(a, b)
end

@testset "workspace setup adds no user construction and stages retain constructor order" begin

    initialized = artifacts(ModelDescription(; type = ConstructedParent,
        models = (;
            a = ModelDescription(; type = ConstructedLeaf, continuous_states = (; x = 0.)),
            b = ModelDescription(; type = ConstructedLeaf, continuous_states = (; x = 10.)),
        )))
    problem = CP.ContinuousProblem(initialized.ommd, (t, model) -> RatesOutput(;
        models = (; a = RatesOutput(; rates = (; x = 1.)))))
    reference = Solvers.step!(persistent_integrator(), problem, step_request(initialized.msd))
    empty!(CONSTRUCTOR_EVENTS)
    integrator = selected_integrator(problem, initialized.msd)
    @test isempty(CONSTRUCTOR_EVENTS)
    result = Solvers.step!(integrator, problem, step_request(initialized.msd))
    @test CONSTRUCTOR_EVENTS == repeat([:leaf, :leaf, :parent], 4)
    @test result.state_at_end.models.a.continuous_states.x ==
        reference.state_at_end.models.a.continuous_states.x
    @test result.state_at_end.models.b === initialized.msd.models.b

end

# This extension is last: it deliberately changes a shared propagation adapter's method
# table, which conservatively disables private workspaces for subsequently created solvers.
struct CopyRetainer
    x::Float64
end
CopyRetainer(; x) = CopyRetainer(x)
const COPIED_DESCRIPTORS = Any[]
function SoS.copy_model_state_description_except(
    state::SoS.ModelStateDescription{CopyRetainer}; kwargs...,
)
    result = invoke(SoS.copy_model_state_description_except,
        Tuple{SoS.ModelStateDescription}, state; kwargs...)
    push!(COPIED_DESCRIPTORS, result)
    return result
end

@testset "custom descriptor copying retains persistent propagation" begin

    initialized = artifacts(ModelDescription(; type = CopyRetainer,
        continuous_states = (; x = 0.)))
    problem = CP.ContinuousProblem(initialized.ommd,
        (t, model) -> RatesOutput(; rates = (; x = 1.)))
    reference_state = artifacts(ModelDescription(; continuous_states = (; x = 0.)))
    reference_problem = CP.ContinuousProblem(reference_state.ommd,
        (t, model) -> RatesOutput(; rates = (; x = 1.)))
    reference = Solvers.step!(persistent_integrator(), reference_problem,
        step_request(reference_state.msd))
    empty!(COPIED_DESCRIPTORS)
    integrator = selected_integrator(problem, initialized.msd)
    @test isempty(COPIED_DESCRIPTORS)
    if HAS_WORKSPACE
        @test integrator.workspace === nothing
    end
    Solvers.step!(integrator, problem, step_request(initialized.msd))
    @test [state.continuous_states.x for state in COPIED_DESCRIPTORS] ==
        [0.125, 0.125, 0.25, reference.state_at_end.continuous_states.x]

end

end # StateWorkspaceTests
