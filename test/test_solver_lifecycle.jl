module TestSolverLifecycle

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

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
    history, t_final, model_final = simulate(
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
            UpdatesOutput()
        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test t_final == 1
    @test model_final.x == 10.
    @test history["/"]["x"].data[end] == 10.
    @test history["/"]["twice_x"].data[end] == 20.

end

@testset "a failed terminal rates sample retains the accepted endpoint" begin

    # The update at t = 1 is already committed when the direct terminal rates evaluation
    # observes x = 10 and throws. The exception must end the simulation without rolling its
    # reported time, final model, or close callback back to the preceding accepted sample.
    close_inputs = Ref{Any}(nothing)
    history, t_final, model_final = @test_logs (:error,) simulate(
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
        close_fcn = (t, model) -> close_inputs[] = (t, model.x),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test t_final == 1
    @test model_final.x == 10.
    @test history.stop isa SystemsOfSystems.EncounteredError
    @test history.stop.time == 1.
    @test close_inputs[] == (1//1, 10.)

end

@testset "intermediate Runge-Kutta stages cannot stop the simulation" begin

    # RK4 evaluates rates at the midpoint of this step twice. Those models are provisional
    # numerical stage models, not accepted simulation samples, so their stop flags must not
    # affect the simulation lifecycle.
    history, t_final, model_final = simulate(
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

    @test t_final == 1
    @test model_final.x ≈ 1.
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

@testset "a rejected attempt cannot stop the simulation" begin

    # Tight tolerances force DP54 to reject its first large attempt. The first beginning
    # evaluation requests a stop, but that evaluation is not an accepted sample. A later
    # attempt at the same official start time is accepted without a stop request, and the
    # simulation must continue normally.
    n_start_evaluations = [0,]
    history, t_final, _ = simulate(
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
    @test t_final == 1
    @test history.stop isa SystemsOfSystems.ReachedEndTime

end

@testset "an accepted rates stop completes its sample" begin

    # A stop request from the authoritative beginning-of-step rates evaluation becomes valid
    # only once the numerical attempt is accepted. The accepted continuous step and its
    # discrete update therefore complete before the sim stops.
    history, t_final, model_final = simulate(
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

    @test t_final == 1//2
    @test model_final.x ≈ 1//2
    @test model_final.n_updates == 1
    @test history.stop isa SystemsOfSystems.ModelRequestedStop

end

@testset "the first model stop request wins deterministically" begin

    # Both child models request a stop in the same accepted RatesOutput. Model hierarchies
    # are traversed parent-first and then depth-first in named-tuple field order, so the
    # request from `first_model` is the one represented in the scalar simulation stop field.
    history, t_final, _ = simulate(
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

    @test t_final == 1//2
    @test history.stop isa SystemsOfSystems.ModelRequestedStop
    @test history.stop.model_path == "/models/first_model"

end

@testset "an update can stop after its accepted sample" begin

    # The update at the accepted endpoint is applied before its stop request takes effect.
    # The direct terminal rates sample then observes and logs that updated state.
    history, t_final, model_final = simulate(
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

    @test t_final == 1//2
    @test model_final.x == 10.
    @test history["/"]["x"].data[end] == 10.
    @test history["/"]["observed_x"].data[end] == 10.
    @test history.stop isa SystemsOfSystems.ModelRequestedStop

end

end # TestSolverLifecycle
