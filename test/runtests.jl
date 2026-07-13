using HDF5Vectors # For the HDF5Logger
using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers, Logs, Hooks
# using GLMakie # For plots

out_dir = "out"
mkpath(joinpath(@__DIR__, out_dir))

include("continuous_random_variables.jl")
include("random_variable_seeds.jl")
include("control_system_demo.jl")
include("test_hooks.jl")
include("test_sim_timeout.jl")
include("test_clock_sync.jl")
include("test_resources.jl")
include("test_updating_continuous_states.jl")

# We implement a custom interpolation type just to test that we can.
struct OffsetLinearInterpolation
    offset::Float64
end

function (interpolator::OffsetLinearInterpolation)(ts, t)
    k_hi = searchsortedfirst(ts.time, t)
    if k_hi == 1
        return interpolator.offset + ts.data[1]
    end
    t_lo = ts.time[k_hi - 1]
    t_hi = ts.time[k_hi]
    y_lo = ts.data[k_hi - 1]
    y_hi = ts.data[k_hi]
    fraction_from_last_to_next = (t - t_lo) / (t_hi - t_lo)
    return interpolator.offset + y_lo + fraction_from_last_to_next * (y_hi - y_lo)
end

struct ConstantInterpolation
    value::Float64
end

function (interpolator::ConstantInterpolation)(ts, t)
    return interpolator.value
end

@testset "is_regular_step_triggering" begin
    @test is_regular_step_triggering(10.1, 0.05) == true
    @test is_regular_step_triggering(10.1, 0.20) == false
    @test is_regular_step_triggering(10.1, 0.) == true # 0 means "always triggering"
    @test is_regular_step_triggering(10.1, 0.20, 0.1) == true
    @test is_regular_step_triggering(10.1, 1., 0.0) == false
    @test is_regular_step_triggering(10.1, 1., 0.1) == true
end

@testset "TimeSeries indexing" begin

    ts = SystemsOfSystems.TimeSeries(;
        title = "Rotor Speed",
        time = collect(0.0:0.1:2.0),
        data = collect(100.0:120.0),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega",
        discrete = false,
    )

    @test ts.interpolator isa SystemsOfSystems.LinearInterpolation
    @test ts[1] == (ts.time[1] => ts.data[1])

    ts_first_10 = ts[1:10]
    @test ts_first_10 isa SystemsOfSystems.TimeSeries
    @test ts_first_10.time == ts.time[1:10]
    @test ts_first_10.data == ts.data[1:10]
    @test ts_first_10.title == ts.title
    @test ts_first_10.time_dimension == ts.time_dimension
    @test ts_first_10.dimensions == ts.dimensions
    @test ts_first_10.path == ts.path
    @test ts_first_10.discrete == ts.discrete
    @test ts_first_10.interpolator isa SystemsOfSystems.LinearInterpolation

    ts_all = ts[:]
    @test ts_all isa SystemsOfSystems.TimeSeries
    @test ts_all.time == ts.time
    @test ts_all.data == ts.data

    @test ts(0.0) == 100.0
    @test ts(0.35) ≈ 103.5
    @test ts(2.0) == 120.0

    ts_resampled = ts(0.0:0.05:0.2)
    @test ts_resampled isa SystemsOfSystems.TimeSeries
    @test ts_resampled.time == collect(0.0:0.05:0.2)
    @test ts_resampled.data ≈ [100.0, 100.5, 101.0, 101.5, 102.0]
    @test ts_resampled.discrete == false
    @test ts_resampled.interpolator isa SystemsOfSystems.LinearInterpolation

    ts_discrete = SystemsOfSystems.TimeSeries(;
        title = "Commanded Speed",
        time = copy(ts.time),
        data = copy(ts.data),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega_cmd",
        discrete = true,
        groups = ts.groups,
    )
    @test ts_discrete.interpolator isa SystemsOfSystems.SampleAndHold
    @test ts_discrete(0.35) == 103.0
    @test ts_discrete(0.4) == 104.0

    ts_discrete_resampled = ts_discrete(0.0:0.05:0.2)
    @test ts_discrete_resampled isa SystemsOfSystems.TimeSeries
    @test ts_discrete_resampled.time == collect(0.0:0.05:0.2)
    @test ts_discrete_resampled.data == [100.0, 100.0, 101.0, 101.0, 102.0]
    @test ts_discrete_resampled.discrete == true
    @test ts_discrete_resampled.interpolator isa SystemsOfSystems.SampleAndHold

    ts_hold = SystemsOfSystems.TimeSeries(;
        title = "Held Rotor Speed",
        time = copy(ts.time),
        data = copy(ts.data),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega_held",
        discrete = false, # Sample-and-hold is the default for discrete, so we're intentionally _not_ allowing it to use the default.
        interpolator = SystemsOfSystems.SampleAndHold,
    )
    @test ts_hold.interpolator isa SystemsOfSystems.SampleAndHold
    @test ts_hold(0.35) == 103.0 # (as opposed to 103.5 for linear)

    ts_discrete_linear = SystemsOfSystems.TimeSeries(;
        title = "Linearly Interpolated Commanded Speed",
        time = copy(ts.time),
        data = copy(ts.data),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega_cmd_linear",
        discrete = true, # Again, we're explicitly changing from the default behavior.
        interpolator = SystemsOfSystems.LinearInterpolation(),
    )
    @test ts_discrete_linear.interpolator isa SystemsOfSystems.LinearInterpolation
    @test ts_discrete_linear(0.35) ≈ 103.5

    offset_interpolator = OffsetLinearInterpolation(10.0)
    ts_custom = SystemsOfSystems.TimeSeries(;
        title = "Offset Rotor Speed",
        time = copy(ts.time),
        data = copy(ts.data),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega_offset",
        interpolator = offset_interpolator,
    )
    @test ts_custom.interpolator === offset_interpolator
    @test ts_custom(0.35) ≈ 113.5
    @test ts_custom(0.0) == 110.0
    @test ts_custom(0.3) == 113.0

    ts_custom_slice = ts_custom[1:5]
    @test ts_custom_slice.interpolator === offset_interpolator

    ts_custom_resampled = ts_custom(0.0:0.05:0.1)
    @test ts_custom_resampled.data ≈ [110.0, 110.5, 111.0]
    @test ts_custom_resampled.interpolator === offset_interpolator

    constant_interpolator = ConstantInterpolation(42.0)
    ts_constant = SystemsOfSystems.TimeSeries(;
        title = "Constant Rotor Speed",
        time = copy(ts.time),
        data = copy(ts.data),
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega_constant",
        interpolator = constant_interpolator,
    )
    @test ts_constant(-100.0) == 42.0
    @test ts_constant(100.0) == 42.0

    show_text = sprint(show, MIME"text/plain"(), ts)
    @test occursin("interpolator:", show_text)

    @test_throws ErrorException ts(-0.01)
    @test_throws ErrorException ts(2.01)

end

@testset "VariableDescription interpolation" begin

    offset_interpolator = OffsetLinearInterpolation(5.0)
    described_state = SystemsOfSystems.VariableDescription(
        0.0;
        title = "Described State",
        dimensions = [SystemsOfSystems.Dimension("state", ""),],
        interpolator = offset_interpolator,
    )
    default_described_state = SystemsOfSystems.VariableDescription(
        0.0;
        title = "Default Described State",
        dimensions = [SystemsOfSystems.Dimension("state", ""),],
    )

    @test described_state.interpolator === offset_interpolator
    @test ismissing(default_described_state.interpolator)

    model_description = SystemsOfSystems.ModelDescription(;
        continuous_states = (;
            x = described_state,
            y = default_described_state,
        ),
    )

    basic_log, basic_history = Logs.create_log(
        Logs.BasicLogOptions(),
        model_description,
        SystemsOfSystems.Dimension("time", "s"),
    )
    @test basic_history.continuous_states.x.interpolator === offset_interpolator
    @test basic_history.continuous_states.y.interpolator isa SystemsOfSystems.LinearInterpolation
    Logs.close_log(basic_log)

    hdf5_log, hdf5_history = Logs.create_log(
        Logs.HDF5LogOptions("$out_dir/variable_description_interpolator.h5"),
        model_description,
        SystemsOfSystems.Dimension("time", "s"),
    )
    @test hdf5_history.continuous_states.x.interpolator === offset_interpolator
    @test hdf5_history.continuous_states.y.interpolator isa SystemsOfSystems.LinearInterpolation
    Logs.close_log(hdf5_log)

end

# This is a continuous-only sim.
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
        Logs.HDF5LogOptions("$out_dir/exponential_logs.h5")
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

    # Test the final state.
    @test t == t_end
    @test model.x ≈ exp(-t_end/time_constant) atol=1e-4

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

        if solver_type == "rk4"
            @test x_ts.time == collect(0. : dt_rk4 : t_end)
        end
    end

    # Check that we can load an HDF5 log and get the same stuff.
    if log_type == "hdf5"
        x_ts = history["/"]["x"]
        @test x_ts.time isa HDF5Vectors.AbstractHDF5Vector
        @test x_ts.data isa HDF5Vectors.AbstractHDF5Vector
        hdf5_log, = Logs.load_hdf5_log("$out_dir/exponential_logs.h5")
        @test collect(history["/"]["x"].time) == collect(hdf5_log["/"]["x"].time)
        @test collect(history["/"]["x"].data) == collect(hdf5_log["/"]["x"].data)
        Logs.close_log(hdf5_log)
    end

    Logs.close_log(history.log)

end

# Here's a discrete-only sim.
@testset failfast = false "discrete exponential" begin

    is_closed = [false,]

    # We'll simulate a pure (and discrete) exponential decay and compare to the known answer.
    time_constant = 2.
    t_end = 5.
    history, t, model = simulate(
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
        close_fcn = (t, model) -> begin
            is_closed[1] = true
        end,
        t = (0, t_end),
    )

    # Test the final state.
    @test t == t_end
    @test model.x ≈ exp(-t_end/time_constant) atol=1e-4

    @test history["/"]["x"].time[1] == 0.
    @test history["/"]["x"].data[1] == 1.
    @test history["/"]["x"].data[end] == model.x
    @test is_closed[1] == true

    x_ts = history["/"]["x"]
    @test x_ts(x_ts.time[1]) == x_ts.data[1]
    t_mid = (x_ts.time[2] + x_ts.time[3]) / 2
    @test x_ts(t_mid) == x_ts.data[2]

    # Test our weird stepping strategy.
    history["/"]["x"].time == vcat(0., collect(0.1 * 1.5^n for n in 0:9), t_end)

end

# Here's a sim with a single hybrid model.
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
                UpdatesOutput()
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

@testset "failed steps in DP54 for max_dt = $max_dt" for max_dt in (10//1, 1//10)

    # This should generate a sinusoid. When the time step is really large, it should fail
    # integration tolerances and end up with smaller steps. When it's really smaller, it
    # should observe the unnecessarily small steps.
    history, t, x = simulate(
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
    history, t, x = simulate(
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

# TODO: Test continuous variables that _don't_ have rates outputs sometimes.
