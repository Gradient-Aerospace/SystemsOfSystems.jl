using HDF5Vectors # For the HDF5Logger
using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers, Logs, Monitors
# using GLMakie # For plots

out_dir = "out"
mkpath(joinpath(@__DIR__, out_dir))

@testset failfast = false "exponential with $solver_type solver, $log_type logs" for solver_type in ("rk4", "dp54"), log_type in ("ram", "hdf5", "null", "nothing")

    dt_rk4 = 0.1
    solver = if solver_type == "dp54"
        Solvers.DormandPrince54Options() # TODO: Test that max_dt limits/doesn't limit.
    elseif solver_type == "rk4"
        Solvers.RungeKutta4Options(; dt = dt_rk4)
    end

    log = if log_type == "ram"
        Logs.BasicLogOptions()
    elseif log_type == "hdf5"
        Logs.HDF5LogOptions("$out_dir/logs.h5")
    elseif log_type == "null"
        Logs.NullLogOptions()
    elseif log_type == "none"
        nothing
    end

    # We'll simulate a pure exponential decay and compare to the known answer.
    time_constant = 2.
    t_end = 5.
    history, t, model = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            constants = (;
                time_constant,
            ),
            continuous_states = (;
                x = 1.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    x = -1/model.time_constant * model.x,
                ),
            )
        end,
        updates_fcn = (t, model) -> UpdatesOutput(),
        t = (0, t_end),
        options = SimOptions(;
            solver,
            log,
            monitors = [Monitors.ProgressBarOptions()],
        ),
    )

    # Test the final state.
    @test t == t_end
    @test model.x ≈ exp(-t_end/time_constant) atol=1e-4

    # We can only test logs when we have logs.
    if log_type == "ram" || log_type == "hdf5"
        @test history["/"]["x"].data[1] == 1.
        @test history["/"]["x"].data[end] == model.x
        if solver_type == "rk4"
            @test history["/"]["x"].time == collect(0. : dt_rk4 : t_end)
        end
    end

    # Check that we can load an HDF5 log and get the same stuff.
    if log_type == "hdf5"
        hdf5_log, = Logs.load_hdf5_log("$out_dir/logs.h5")
        @test collect(history["/"]["x"].time) == collect(hdf5_log["/"]["x"].time)
        @test collect(history["/"]["x"].data) == collect(hdf5_log["/"]["x"].data)
        Logs.close_log(hdf5_log)
    end

    Logs.close_log(history.log)

end

@testset "closed loop control" begin

    # We'll simulate a closed-loop control system to test hybrid systems.
    dt = 0.05
    t_end = 5.
    history, t, x = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            constants = (;
                dt = dt,
                kp = 8.,
                kd = 4.,
                mass = 1.,
            ),
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
            continuous_outputs = (;
                acceleration = 0.,
            ),
            discrete_states = (;
                force = 0.,
            ),
            discrete_outputs = (;
                control_error = 0.,
            ),
            t_next = 0.05,
        ),
        rates_fcn = (t, model) -> begin
            acceleration = model.force / model.mass
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = acceleration,
                ),
                outputs = (;
                    acceleration,
                )
            )
        end,
        updates_fcn = (t, model) -> begin
            if is_regular_step_triggering(t, model.dt)
                UpdatesOutput(
                    updates = (;
                        force = -model.kp * model.position - model.kd * model.velocity,
                    ),
                    outputs = (;
                        control_error = model.position,
                    ),
                    t_next = t + model.dt,
                )
            else
                UpdatesOutput()
            end
        end,
        t = (0, t_end),
    )

    # It started in the right place.
    @test history["/"]["position"].data[1] == 1.

    # THe control system more or less worked.
    @test abs(history["/"]["position"].data[end]) < 0.1

    # We got the expected number of discrete steps.
    @test length(history["/"]["force"].data) == t_end / dt + 1
    @test length(history["/"]["control_error"].data) == t_end / dt + 1

end
