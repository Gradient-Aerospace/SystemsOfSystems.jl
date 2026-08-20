module TestTimeSeries

using Test
using HDF5Vectors # For the HDF5Logger
using SystemsOfSystems
using SystemsOfSystems: Logs

const out_dir = joinpath(@__DIR__, "out")
mkpath(out_dir)

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

@testset "TimeSeries selection" begin

    source_interpolator = OffsetLinearInterpolation(10.0)
    ts = SystemsOfSystems.TimeSeries(;
        title = "Rotor Speed",
        time = [0.0, 0.1, 0.2],
        data = [100.0, 110.0, 120.0],
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = [SystemsOfSystems.Dimension("angular speed", "rad/s"),],
        path = "/rotor/omega",
        discrete = false,
        interpolator = source_interpolator,
    )

    # A selection transforms each native data value while retaining the temporal metadata.
    # Dimensions, groups, and interpolation are inferred for the transformed values.
    selected = SystemsOfSystems.select(ts) do speed
        speed^2
    end
    @test selected.data == ts.data .^ 2
    @test selected.time == ts.time
    # Check that they are different array objects in memory.
    @test selected.time !== ts.time
    @test selected.title == ts.title
    @test selected.time_dimension == ts.time_dimension
    @test selected.path == ts.path
    @test selected.discrete == ts.discrete
    @test selected.dimensions == [SystemsOfSystems.Dimension("1", ""),]
    @test selected.groups == ["1" => ["1",],]
    @test selected.interpolator isa SystemsOfSystems.LinearInterpolation
    @test selected.interpolator !== source_interpolator
    @test ts.data == [100.0, 110.0, 120.0]
    @test ts.dimensions == [SystemsOfSystems.Dimension("angular speed", "rad/s"),]

    # Callers can replace all result metadata that is not intrinsically tied to time.
    selected_dimension = SystemsOfSystems.Dimension("scaled speed", "rad/s")
    selected_groups = ["Speed" => ["scaled speed",],]
    selected_interpolator = ConstantInterpolation(42.0)
    selected_with_metadata = SystemsOfSystems.select(
        identity,
        ts;
        title = "Scaled Rotor Speed",
        dimensions = [selected_dimension,],
        path = "/rotor/scaled_omega",
        discrete = true,
        interpolator = selected_interpolator,
        groups = selected_groups,
    )
    @test selected_with_metadata.title == "Scaled Rotor Speed"
    @test selected_with_metadata.dimensions == [selected_dimension,]
    @test selected_with_metadata.path == "/rotor/scaled_omega"
    @test selected_with_metadata.discrete
    @test selected_with_metadata.interpolator === selected_interpolator
    @test selected_with_metadata.groups == selected_groups
    @test selected_with_metadata.time_dimension == ts.time_dimension

    # Typed empty data remains typed after a transformation whose result Julia can infer.
    empty_ts = SystemsOfSystems.TimeSeries(;
        title = "Empty Rotor Speed",
        time = Float64[],
        data = Float64[],
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        path = "/rotor/empty_omega",
    )
    empty_selected = SystemsOfSystems.select(speed -> 2.0 * speed, empty_ts)
    @test isempty(empty_selected.data)
    @test eltype(empty_selected.data) == Float64

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
        Logs.HDF5LogOptions(joinpath(out_dir, "variable_description_interpolator.h5")),
        model_description,
        SystemsOfSystems.Dimension("time", "s"),
    )
    @test hdf5_history.continuous_states.x.interpolator === offset_interpolator
    @test hdf5_history.continuous_states.y.interpolator isa SystemsOfSystems.LinearInterpolation
    Logs.close_log(hdf5_log)

end

end # TestTimeSeries
