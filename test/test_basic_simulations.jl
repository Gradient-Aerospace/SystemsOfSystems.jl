module TestBasicSimulations

using Test
using HDF5Vectors # For the HDF5Logger
using SystemsOfSystems
using SystemsOfSystems: Solvers, Logs, Hooks
# using GLMakie # For plots

const out_dir = joinpath(@__DIR__, "out")
mkpath(out_dir)

@testset "log display" begin

    # Property destructuring returns the selected property in the REPL. The compact show
    # method must therefore summarize the log instead of displaying its full backing
    # dictionary and every stored sample.
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(),
    )
    (; log) = history

    @test sprint(show, log) == "BasicLog with 1 model history"
    @test sprint(show, MIME"text/plain"(), log) == "Model Histories:\n  /\n"

end

# This is a continuous-only sim.
@testset failfast = false "exponential with $solver_type solver, $log_type logs" for solver_type in ("rk2", "rk4", "dp54"), log_type in ("ram", "hdf5", "null", "nothing")

    fixed_dt = 0.1
    solver = if solver_type == "dp54"
        Solvers.DormandPrince54Options()
    elseif solver_type == "rk2"
        Solvers.Ralston2Options(; dt = fixed_dt)
    elseif solver_type == "rk4"
        Solvers.RungeKutta4Options(; dt = fixed_dt)
    end

    log = if log_type == "ram"
        Logs.BasicLogOptions()
    elseif log_type == "hdf5"
        Logs.HDF5LogOptions(joinpath(out_dir, "exponential_logs.h5"))
    elseif log_type == "null"
        Logs.NullLogOptions()
    elseif log_type == "none"
        nothing
    end

    # We'll simulate a pure exponential decay and compare to the known answer.
    time_constant = 2.
    t_end = 5.
    history = simulate(
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
            if !isa(t, Float64)
                error("The time input to rates should be a Float64.")
            end
            RatesOutput(
                rates = (;
                    x = -1/model.time_constant * model.x,
                ),
            )
        end,
        t = (0, t_end),
        options = SimOptions(;
            solver,
            log,
            hooks = [Hooks.ProgressBarOptions()],
        ),
    )
    (; t_stop, model) = history

    # Test the final state.
    @test history.t_start == 0
    @test t_stop == t_end
    @test model.x ≈ exp(-t_end/time_constant) atol=1e-4
    @test succeeded(history)
    @test fieldtype(typeof(history), :model) == typeof(model)

    # We can only test logs when we have logs.
    if log_type == "ram" || log_type == "hdf5"
        x_ts = history["/"]["x"]
        x_first = x_ts[1]
        @test x_first == (x_ts.time[1] => 1.)
        @test x_ts.data[end] == model.x

        # New accessors should work for both in-memory and HDF5-backed vectors.
        idxs = 1:min(3, length(x_ts.time))
        x_slice = x_ts[idxs]
        @test x_slice isa SystemsOfSystems.TimeSeries
        @test collect(x_slice.time) == [x_ts.time[k] for k in idxs]
        @test collect(x_slice.data) == [x_ts.data[k] for k in idxs]
        @test x_slice.interpolator isa SystemsOfSystems.LinearInterpolation

        # Exact-time access returns the exact stored sample.
        @test x_ts(x_ts.time[1]) == x_ts.data[1]

        # Between-sample access is linear interpolation for continuous series.
        if length(x_ts.time) >= 2 && x_ts.time[1] != x_ts.time[2]
            t1, t2 = x_ts.time[1], x_ts.time[2]
            x1, x2 = x_ts.data[1], x_ts.data[2]
            t_mid = (t1 + t2) / 2
            expected = x1 + (t_mid - t1) / (t2 - t1) * (x2 - x1)
            @test x_ts(t_mid) ≈ expected
        end

        if solver_type == "rk2" || solver_type == "rk4"
            @test x_ts.time == collect(0. : fixed_dt : t_end)
        end
    end

    # Check that we can load an HDF5 log and get the same stuff.
    if log_type == "hdf5"
        x_ts = history["/"]["x"]
        @test x_ts.time isa HDF5Vectors.AbstractHDF5Vector
        @test x_ts.data isa HDF5Vectors.AbstractHDF5Vector
        hdf5_log, = Logs.load_hdf5_log(joinpath(out_dir, "exponential_logs.h5"))
        @test collect(history["/"]["x"].time) == collect(hdf5_log["/"]["x"].time)
        @test collect(history["/"]["x"].data) == collect(hdf5_log["/"]["x"].data)
        Logs.close_log(hdf5_log)
    end

    Logs.close_log(history.log)

end

@testset "$label convergence" for (label, make_solver, expected_range) in (
    ("Ralston second-order", dt -> Solvers.Ralston2Options(; dt), (3.5, 4.5)),
    ("Runge-Kutta fourth-order", dt -> Solvers.RungeKutta4Options(; dt), (14., 17.)),
)

    # The exact solution of x' = x with x(0) = 1 is exp(t). Halving a fixed step should
    # reduce global error by approximately 2^p for an order-p method.
    function final_error(dt)
        history = simulate(
            nothing;
            t = (0, 1),
            init_fcn = (args...) -> ModelDescription(;
                continuous_states = (; x = 1.),
            ),
            rates_fcn = (t, model) -> RatesOutput(;
                rates = (; x = model.x,),
            ),
            options = SimOptions(;
                log = nothing,
                solver = make_solver(dt),
            ),
        )
        return abs(history.model.x - exp(1.))
    end

    coarse_error = final_error(1//10)
    fine_error = final_error(1//20)
    error_reduction = coarse_error / fine_error
    @test first(expected_range) < error_reduction < last(expected_range)

end

# Here's a discrete-only sim.
@testset failfast = false "discrete exponential" begin

    # We'll simulate a pure (and discrete) exponential decay and compare to the known answer.
    time_constant = 2.
    t_end = 5.
    history = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            discrete_states = (;
                x = 1.,
                t = 0.,
            ),
            t_next = 0.1,
        ),
        updates_fcn = (t, model) -> UpdatesOutput(;
            updates = (;
                t = t,
                x = exp(-(t - model.t)/time_constant) * model.x
            ),
            t_next = 1.5 * t, # Just for fun, steps change size.
        ),
        t = (0, t_end),
    )
    (; t_stop, model) = history

    # Test the final state.
    @test t_stop == t_end
    @test model.x ≈ exp(-t_end/time_constant) atol=1e-4

    @test history["/"]["x"].time[1] == 0.
    @test history["/"]["x"].data[1] == 1.
    @test history["/"]["x"].data[end] == model.x

    x_ts = history["/"]["x"]
    @test x_ts(x_ts.time[1]) == x_ts.data[1]
    t_mid = (x_ts.time[2] + x_ts.time[3]) / 2
    @test x_ts(t_mid) == x_ts.data[2]

    # Test our weird stepping strategy.
    @test history["/"]["x"].time ≈
        vcat(0., collect(0.1 * 1.5^n for n in 0:9), t_end)

end

# Here's a sim with a single hybrid model.
@testset "closed loop control" begin

    # We'll simulate a closed-loop control system to test hybrid systems.
    dt = 0.05
    t_end = 5.
    history = simulate(
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
                # This tests the "missing" is an acceptable initial value (and not logged).
                # We must include a full VariableDescription here so that we can give it a
                # type up front (the type parameter), since that obvioulsy can't be inferred
                # from the value (missing) itself.
                control_error = VariableDescription{Float64}(
                    missing;
                    title = "Control Error",
                    dimensions = ["error" => "m",],
                ),
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
                nothing
            end
        end,
        t = (0, t_end),
    )

    # It started in the right place.
    @test history["/"]["position"].data[1] == 1.

    # The control system more or less worked.
    @test abs(history["/"]["position"].data[end]) < 0.1

    # We got the expected number of discrete steps.
    @test length(history["/"]["force"].data) == t_end / dt + 1
    @test length(history["/"]["control_error"].data) == t_end / dt # Ignores missing 1st el.
    @test eltype(history["/"]["control_error"].data) == Float64

end

@testset "omitted continuous rates preserve states" begin

    # Continuous states and submodels without active dynamics may be omitted from a
    # RatesOutput. A discrete state changes the rates structure only between accepted
    # steps, keeping every Runge-Kutta attempt internally consistent.
    history = simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0., y = 0.),
            discrete_states = (; auxiliary_dynamics_active = true,),
            models = (;
                child = ModelDescription(;
                    continuous_states = (; z = 0.,),
                ),
            ),
        ),
        rates_fcn = (t, model) -> if model.auxiliary_dynamics_active
            RatesOutput(;
                rates = (; x = 1., y = 2.),
                models = (;
                    child = RatesOutput(; rates = (; z = 3.,)),
                ),
            )
        else
            RatesOutput(; rates = (; x = 1.,))
        end,
        updates_fcn = (t, model) -> if t == 1
            UpdatesOutput(; updates = (; auxiliary_dynamics_active = false,),)
        else
            nothing
        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    # All three states propagate over the first second. Thereafter x continues, while the
    # omitted y derivative and child RatesOutput leave their states unchanged.
    @test history.model.x ≈ 2.
    @test history.model.y ≈ 2.
    @test history.model.child.z ≈ 3.
    @test history["/"]["x"].data ≈ [0., 1., 2.]
    @test history["/"]["y"].data ≈ [0., 2., 2.]
    @test history["/child"]["z"].data ≈ [0., 3., 3.]

end

end # TestBasicSimulations
