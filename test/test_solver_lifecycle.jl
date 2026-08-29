module TestSolverLifecycle

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

struct UnconvergeableState
    value::Float64
end

Base.:+(a::UnconvergeableState, b::UnconvergeableState) =
    UnconvergeableState(a.value + b.value)
Base.:*(scale::Number, state::UnconvergeableState) =
    UnconvergeableState(scale * state.value)

function SystemsOfSystems.normalized_variable_error(
    value::UnconvergeableState,
    embedded_value::UnconvergeableState,
    absolute_tolerance,
    relative_tolerance,
)
    return 2.
end

# These tests describe the boundary between the continuous solver and the hybrid simulation
# loop. In particular, a solver step is not merely an internal numerical detail: every
# accepted step is followed by hooks, discrete random draws, and the model's discrete
# update. Keeping that lifecycle explicit prevents a future solver backend from silently
# integrating over several externally visible sample times in one call.

@testset "terminal updates are followed by a real rates sample" begin

    # The terminal update deliberately changes a continuous state. The final continuous
    # state and output in the log must both describe the post-update model. Historically,
    # the sim obtained this output by asking the solver to take a zero-duration step; the
    # desired interface samples rates directly instead.
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; twice_x = 0.),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.),
            outputs = (; twice_x = 2 * model.x,),
        ),
        updates_fcn = (t, model) -> if t == 1
            UpdatesOutput(; updates = (; x = 10.,),)
        else
            nothing
        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test history.t_stop == 1
    @test history.model.x == 10.
    @test history["/"]["x"].data[end] == 10.
    @test history["/"]["twice_x"].data[end] == 20.

end

@testset "a failed terminal rates sample retains the accepted endpoint" begin

    # The update at t = 1 is already committed when the direct terminal rates evaluation
    # observes x = 10 and throws. The exception must end the simulation without rolling its
    # reported time or final model back to the preceding accepted sample.
    history = @test_logs (:error,) simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
        ),
        rates_fcn = (t, model) -> begin

            if t == 1. && model.x == 10.
                error("The terminal rates sample failed.")
            end

            return RatesOutput(; rates = (; x = 1.,),)

        end,
        updates_fcn = (t, model) -> UpdatesOutput(;
            updates = t == 1 ? (; x = 10.,) : (;),
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test history.t_stop == 1
    @test history.model.x == 10.
    @test history.stop isa SystemsOfSystems.EncounteredError
    @test history.stop.time == 1.
    @test history["/"]["x"].time[end] == 1.
    @test history["/"]["x"].data[end] == 10.
    @test !succeeded(history)

end

@testset "a failed next rates sample retains the propagated state" begin

    # The first step propagates x from t = 0 to t = 1 and commits that endpoint. The rates
    # evaluation beginning the next step then fails. The state at t = 1 is known and should
    # remain in the log, while no corresponding output exists because rates never returned.
    n_rates_at_one = Ref(0)
    history = @test_logs (:error,) simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; observed_x = 0.),
        ),
        rates_fcn = (t, model) -> begin

            if t == 1.
                n_rates_at_one[] += 1
                if n_rates_at_one[] == 2
                    error("The next rates sample failed.")
                end
            end

            return RatesOutput(;
                rates = (; x = 1.,),
                outputs = (; observed_x = model.x,),
            )

        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    x_history = history["/"]["x"]
    output_history = history["/"]["observed_x"]
    @test history.t_stop == 1
    @test history.model.x ≈ 1.
    @test history.stop isa SystemsOfSystems.EncounteredError
    @test x_history.time[end] == 1.
    @test x_history.data[end] == history.model.x
    @test output_history.time[end] == 0.

end

@testset "a solver-reported failure retains the propagated state" begin

    # At this large epoch, adjacent integer times have the same Float64 representation.
    # The first one-second interval is nevertheless a hard user boundary and succeeds. The
    # following soft one-second proposal cannot advance floating-point solver time, so the
    # solver reports underflow. The state committed at the first boundary must remain logged.
    t_start = 2^60
    history = simulate(
        nothing;
        t = (t_start, t_start + 1, t_start + 3),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.,),
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    x_history = history["/"]["x"]
    @test history.t_stop == t_start + 1
    @test history.model.x ≈ 1.
    @test history.stop isa Solvers.SolverStepSizeUnderflow
    @test length(x_history.time) == 2
    @test x_history.data[end] == history.model.x

end

@testset "Dormand-Prince reports failure after exhausting rejected attempts" begin

    # A user-defined error policy can reject every numerical attempt. This deterministic
    # policy exercises the retry limit without relying on floating-point underflow or a
    # specially tuned differential equation.
    initial_state = UnconvergeableState(0.)
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = initial_state,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = UnconvergeableState(1.),),
        ),
        options = SimOptions(;
            log = nothing,
            solver = Solvers.DormandPrince54Options(;
                initial_dt = 1,
                max_dt = 1,
                abs_tol = 1.,
                rel_tol = 0.,
            ),
        ),
    )

    @test history.t_stop == 0
    @test history.model.x == initial_state
    @test history.stop isa Solvers.SolverFailedToConverge
    @test !succeeded(history)
    @test SystemsOfSystems.describe(history.stop) ==
        "The solver failed to converge at time 0.0."

end

@testset "intermediate Runge-Kutta stages cannot stop the simulation" begin

    # RK4 evaluates rates at the midpoint of this step twice. Those models are provisional
    # numerical stage models, not accepted simulation samples, so their stop flags must not
    # affect the simulation lifecycle.
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.),
            stop = t == 0.5,
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test history.t_stop == 1
    @test history.model.x ≈ 1.
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

@testset "a rejected attempt cannot stop the simulation" begin

    # Tight tolerances force DP54 to reject its first large attempt. The first beginning
    # evaluation requests a stop, but that evaluation is not an accepted sample. A later
    # attempt at the same official start time is accepted without a stop request, and the
    # simulation must continue normally.
    n_start_evaluations = [0,]
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 1.),
        ),
        rates_fcn = (t, model) -> begin

            if t == 0.
                n_start_evaluations[1] += 1
            end

            return RatesOutput(;
                rates = (; x = -model.x,),
                stop = t == 0. && n_start_evaluations[1] == 1,
            )

        end,
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                initial_dt = 1,
                max_dt = 1,
                abs_tol = 1e-10,
                rel_tol = 1e-10,
            ),
        ),
    )

    @test n_start_evaluations[1] > 1
    @test history.t_stop == 1
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

@testset "an accepted rates stop completes its sample" begin

    # A stop request from the authoritative beginning-of-step rates evaluation becomes valid
    # only once the numerical attempt is accepted. The accepted continuous step and its
    # discrete update therefore complete before the sim stops.
    history = simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            discrete_states = (; n_updates = 0,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.),
            stop = t == 0.,
        ),
        updates_fcn = (t, model) -> UpdatesOutput(;
            updates = (; n_updates = model.n_updates + 1,),
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1//2),
        ),
    )

    @test history.t_stop == 1//2
    @test history.model.x ≈ 1//2
    @test history.model.n_updates == 1
    @test history.stop isa SystemsOfSystems.ModelRequestedStop
    @test succeeded(history)

end

@testset "the first model stop request wins deterministically" begin

    # Both child models request a stop in the same accepted RatesOutput. Model hierarchies
    # are traversed parent-first and then depth-first in named-tuple field order, so the
    # request from `first_model` is the one represented in the scalar simulation stop field.
    history = simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            models = (;
                first_model = ModelDescription(),
                second_model = ModelDescription(),
            ),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            models = (;
                first_model = RatesOutput(; stop = t == 0.,),
                second_model = RatesOutput(; stop = t == 0.,),
            ),
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1//2),
        ),
    )

    @test history.t_stop == 1//2
    @test history.stop isa SystemsOfSystems.ModelRequestedStop
    @test history.stop.model_path == "/models/first_model"

end

@testset "nested model stop paths retain their separators" begin

    # A stop path is assembled while the recursive traversal unwinds. Every model level
    # must retain the separator between its name and the next `/models` path component.
    output = RatesOutput(;
        models = (;
            outer_model = RatesOutput(;
                models = (;
                    inner_model = RatesOutput(; stop = true,),
                ),
            ),
        ),
    )

    stop = SystemsOfSystems.find_model_requested_stop(output)

    @test stop isa SystemsOfSystems.ModelRequestedStop
    @test stop.model_path == "/models/outer_model/models/inner_model"

end

@testset "stop traversal handles wide model hierarchies" begin

    # Wide named tuples caused the recursive Base.tail implementation to specialize on a
    # long chain of successively shorter tuple types. Put the only stop in the final field
    # so the generated implementation must visit every emitted child expression.
    n_models = 64
    names = ntuple(index -> Symbol("model_$index"), Val(n_models))
    models = NamedTuple{names}(
        ntuple(Val(n_models)) do index
            RatesOutput(; stop = index == n_models)
        end,
    )

    stop = @inferred(
        Union{Nothing, SystemsOfSystems.ModelRequestedStop},
        SystemsOfSystems.find_model_requested_stop(
            RatesOutput(; models),
        ),
    )

    @test stop isa SystemsOfSystems.ModelRequestedStop
    @test stop.model_path == "/models/model_64"

end

@testset "an update can stop after its accepted sample" begin

    # The update at the accepted endpoint is applied before its stop request takes effect.
    # The direct terminal rates sample then observes and logs that updated state.
    history = simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; observed_x = 0.),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.),
            outputs = (; observed_x = model.x,),
        ),
        updates_fcn = (t, model) -> UpdatesOutput(;
            updates = t == 1//2 ? (; x = 10.,) : (;),
            stop = t == 1//2,
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1//2),
        ),
    )

    @test history.t_stop == 1//2
    @test history.model.x == 10.
    @test history["/"]["x"].data[end] == 10.
    @test history["/"]["observed_x"].data[end] == 10.
    @test history.stop isa SystemsOfSystems.ModelRequestedStop

end

end # TestSolverLifecycle
