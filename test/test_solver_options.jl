module TestSolverOptions

using StaticArrays: SVector
using Test
using SystemsOfSystems
using SystemsOfSystems: ContinuousProblems, Solvers

struct OscillatorState
    position::Float64
    velocity::Float64
end

function Base.:+(a::OscillatorState, b::OscillatorState)
    return OscillatorState(
        a.position + b.position,
        a.velocity + b.velocity,
    )
end

function Base.:*(scale::Number, state::OscillatorState)
    return OscillatorState(
        scale * state.position,
        scale * state.velocity,
    )
end

function SystemsOfSystems.normalized_variable_error(
    value::OscillatorState,
    embedded_value::OscillatorState,
    absolute_tolerance,
    relative_tolerance,
)
    return max(
        normalized_scalar_error(
            value.position,
            embedded_value.position,
            absolute_tolerance,
            relative_tolerance,
        ),
        normalized_scalar_error(
            value.velocity,
            embedded_value.velocity,
            absolute_tolerance,
            relative_tolerance,
        ),
    )
end

oscillator_components(state::SVector{2, Float64}) = Tuple(state)
oscillator_components(state::OscillatorState) = (state.position, state.velocity)

struct RecordingSolverOptions{O} <: Solvers.AbstractSolverOptions
    solver::O
    step_count::Base.RefValue{Int}
end

struct RecordingIntegrator{I} <: Solvers.AbstractIntegrator
    integrator::I
    step_count::Base.RefValue{Int}
end

function Solvers.create_integrator(
    options::RecordingSolverOptions,
    problem,
    initial_state,
)
    integrator = Solvers.create_integrator(options.solver, problem, initial_state)
    return RecordingIntegrator(integrator, options.step_count)
end

function Solvers.step!(integrator::RecordingIntegrator, problem, request)
    integrator.step_count[] += 1
    return Solvers.step!(integrator.integrator, problem, request)
end

@testset "wide model propagation" begin

    # A wide, flat model hierarchy is where recursive Base.tail tuple processing becomes
    # expensive for the compiler. Use enough fields to exercise the generated expansion
    # while keeping the numerical result simple enough to verify field by field.
    n_models = 64
    names = ntuple(index -> Symbol("model_$index"), Val(n_models))
    submodels = NamedTuple{names}(
        ntuple(Val(n_models)) do index
            SystemsOfSystems.ModelStateDescription{Nothing}(;
                continuous_states = (; x = Float64(index)),
            )
        end,
    )

    # The two stages deliberately use different values so this verifies that the generated
    # code selects the matching field from every stage, rather than accidentally reusing
    # one model's rates.
    function make_stage_rates(multiplier)
        return NamedTuple{names}(
            ntuple(Val(n_models)) do index
                RatesOutput(; rates = (; x = multiplier * index))
            end,
        )
    end

    # RatesOutputs may omit submodels without continuous dynamics. Omit the final submodel
    # from every stage to verify that the generated code supplies the same empty RatesOutput
    # that the previous complete_model_rates path supplied.
    function omit_final_model(stage)
        return NamedTuple{Base.front(names)}(Base.front(Tuple(stage)))
    end
    gains = (1/4, 1/2)
    first_stage = omit_final_model(make_stage_rates(1.))
    second_stage = omit_final_model(make_stage_rates(2.))
    model_rates_at_stages = (first_stage, second_stage)

    propagated = @inferred ContinuousProblems.propagate_models(
        submodels,
        gains,
        model_rates_at_stages,
    )

    @test keys(propagated) == names
    for index in 1:n_models
        initial_value = Float64(index)
        expected_value = initial_value
        if index < n_models
            expected_value += gains[1] * index + gains[2] * 2 * index
        end
        @test propagated[index].continuous_states.x == expected_value
    end

end

@testset "user-defined solver interface" begin

    # A user-defined solver only needs to supply options, a runtime integrator, and the two
    # solver protocol methods. This small wrapper makes those calls observable while
    # delegating the numerical method itself to the built-in RK4 integrator.
    step_count = Ref(0)
    solver = RecordingSolverOptions(
        Solvers.RungeKutta4Options(; dt = 1//4),
        step_count,
    )
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 2.,),
        ),
        options = SimOptions(; solver),
    )

    @test step_count[] == 4
    @test history["/"]["x"].time == collect(0. : 0.25 : 1.)
    @test history.model.x ≈ 2.
    @test history.t_stop == 1

end

@testset "structured continuous state using $label" for (label, make_state) in (
    ("SVector", (position, velocity) -> SVector{2, Float64}(position, velocity)),
    ("custom struct", OscillatorState),
)

    initial_state = make_state(1., 0.)
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (;
                oscillator = VariableDescription(
                    initial_state;
                    title = "Oscillator State",
                    dimensions = ["position" => "m", "velocity" => "m/s"],
                ),
            ),
        ),
        rates_fcn = (t, model) -> begin

            position, velocity = oscillator_components(model.oscillator)
            return RatesOutput(;
                rates = (;
                    oscillator = make_state(velocity, -position),
                ),
            )

        end,
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                abs_tol = 1e-9,
                rel_tol = 1e-9,
            ),
        ),
    )

    # Both representations follow the same harmonic-oscillator solution while retaining
    # their concrete type through propagation and logging.
    position, velocity = oscillator_components(history.model.oscillator)
    @test position ≈ cos(1.) atol = 1e-8
    @test velocity ≈ -sin(1.) atol = 1e-8
    @test history.model.oscillator isa typeof(initial_state)
    @test all(value isa typeof(initial_state) for value in history["/"]["oscillator"].data)

end

@testset "Dormand-Prince maximum step size" begin

    function run_constant_rate(max_dt)

        return simulate(
            nothing;
            t = (0, 2),
            init_fcn = (args...) -> ModelDescription(;
                continuous_states = (; x = 0.,),
            ),
            rates_fcn = (t, model) -> RatesOutput(;
                rates = (; x = 1.,),
            ),
            options = SimOptions(;
                solver = Solvers.DormandPrince54Options(;
                    initial_dt = 1//4,
                    max_dt,
                ),
            ),
        )

    end

    # With no integration error, the adaptive controller grows its step after the first
    # quarter second. A large finite limit should therefore behave exactly like no limit.
    uncapped = run_constant_rate(1//0)
    large_cap = run_constant_rate(10//1)
    @test uncapped["/"]["x"].time == [0., 0.25, 2.]
    @test large_cap["/"]["x"].time == uncapped["/"]["x"].time
    @test large_cap.model.x ≈ uncapped.model.x ≈ 2.

    # A half-second limit constrains every step after the initial quarter-second request;
    # the final shortened step lands exactly on the requested end time.
    capped = run_constant_rate(1//2)
    @test capped["/"]["x"].time == [0., 0.25, 0.75, 1.25, 1.75, 2.]
    @test capped.model.x ≈ 2.

end

@testset "failed steps in DP54 for max_dt = $max_dt" for max_dt in (10//1, 1//10)

    # This should generate a sinusoid. When the time step is really large, it should fail
    # integration tolerances and end up with smaller steps. When it's really smaller, it
    # should observe the unnecessarily small steps.
    history = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = -model.position,
                ),
            )
        end,
        t = (0, 30),
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                initial_dt = max_dt, # Intentionally too big, to make sure this fails.
                max_dt, # Intentionally smaller than necessary.
            ),
        ),
    )

    # Make sure we're always less than the maximum.
    t = history["/"]["position"].time
    for k in 2:length(t)
        @test t[k] - t[k-1] - eps(t[k]) <= max_dt
    end

end

@testset "user-specified time steps" begin

    # This should generate a sinusoid. The max step will limit the step size here, but we'll
    # request even shorter time steps for the first several steps.
    max_dt = 1//2
    t_specified = [0.01, 0.1, 0.2, 0.3, 30.] # Includes a non-zero start
    history = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = -model.position,
                ),
            )
        end,
        t = t_specified,
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                initial_dt = max_dt, # Intentionally too big, to make sure this fails.
                max_dt, # Intentionally smaller than necessary.
            ),
        ),
    )

    # Make sure we're always less than the maximum.
    t = history["/"]["position"].time
    for k in 2:length(t)
        @test t[k] - t[k-1] - eps(t[k]) <= max_dt
    end

    # Make sure the first several steps are precisely what we specified.
    @test t[1:4] == t_specified[1:4]
    @test t[end] == t_specified[end]

end

end # TestSolverOptions
