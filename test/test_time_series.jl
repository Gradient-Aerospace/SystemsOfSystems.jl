module TestTimeSeries

import Dimensions
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

# This lets us exercise SystemsOfSystems' unsupported-constant handling without depending
# on the implementation details of HDF5Vectors' built-in type support.
struct UnsupportedHDF5Constant end

function HDF5Vectors.storage_style(::Type{UnsupportedHDF5Constant}; kwargs...)
    error("Intentional unsupported constant for testing.")
end

struct IMUMeasurement
    accelerometer::Float64
    angular_rate::Float64
end
Dimensions.dimstyle(::Type{IMUMeasurement}) = Dimensions.StructDimensionStyle()

struct HeterogeneousMeasurement
    temperature::Float64
    valid::Bool
end
Dimensions.dimstyle(::Type{HeterogeneousMeasurement}) =
    Dimensions.StructDimensionStyle()

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

    # A selection function can access fields of each native measurement.
    measurements = [
        IMUMeasurement(1.0, 0.1),
        IMUMeasurement(2.0, 0.2),
        IMUMeasurement(3.0, 0.3),
    ]
    measurements_ts = SystemsOfSystems.TimeSeries(;
        title = "IMU Measurements",
        time = [0.0, 0.1, 0.2],
        data = measurements,
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = ["acceleration" => "m/s^2", "angular rate" => "rad/s"],
        path = "/imu/measurement",
        discrete = true,
    )
    accelerometer_ts = SystemsOfSystems.select(measurements_ts) do measurement
        measurement.accelerometer
    end
    @test accelerometer_ts.data == [1.0, 2.0, 3.0]
    @test accelerometer_ts.time == measurements_ts.time
    @test accelerometer_ts.time_dimension == measurements_ts.time_dimension
    @test accelerometer_ts.path == measurements_ts.path
    @test accelerometer_ts.discrete == measurements_ts.discrete
    @test accelerometer_ts.dimensions == [SystemsOfSystems.Dimension("1", ""),]
    @test accelerometer_ts.groups == ["1" => ["1",],]
    @test accelerometer_ts.interpolator isa SystemsOfSystems.SampleAndHold
    @test measurements_ts.data == measurements

    # Dimension selection follows the same flattened representation used by Dimensions.
    acceleration_ts = Dimensions.getdim(measurements_ts, 1)
    angular_rate_ts = Dimensions.getdim(measurements_ts, 2)
    @test acceleration_ts.data == [1.0, 2.0, 3.0]
    @test angular_rate_ts.data == [0.1, 0.2, 0.3]
    @test acceleration_ts.dimensions == [measurements_ts.dimensions[1],]
    @test angular_rate_ts.dimensions == [measurements_ts.dimensions[2],]

    # Dimension selection identifies the dimension while preserving source metadata.
    @test acceleration_ts.title == "IMU Measurements, dimension = 1"
    @test angular_rate_ts.title == "IMU Measurements, dimension = 2"
    @test acceleration_ts.time == measurements_ts.time
    @test acceleration_ts.time_dimension == measurements_ts.time_dimension
    @test acceleration_ts.path == measurements_ts.path
    @test acceleration_ts.discrete == measurements_ts.discrete

    # Grouping and interpolation are inferred for the scalar result.
    @test acceleration_ts.groups == ["acceleration" => ["acceleration",],]
    @test acceleration_ts.interpolator isa SystemsOfSystems.SampleAndHold

    # Callers can override the metadata supplied by dimension selection.
    selected_dimension = SystemsOfSystems.Dimension("selected acceleration", "m/s^2")
    selected_ts = Dimensions.getdim(
        measurements_ts,
        1;
        title = "Selected Acceleration",
        dimensions = [selected_dimension,],
        path = "/imu/selected_acceleration",
    )
    @test selected_ts.title == "Selected Acceleration"
    @test selected_ts.dimensions == [selected_dimension,]
    @test selected_ts.path == "/imu/selected_acceleration"

    # Dimensions can also be selected by their metadata labels.
    acceleration_by_name_ts = SystemsOfSystems.select(measurements_ts, "acceleration")
    angular_rate_by_name_ts = SystemsOfSystems.select(measurements_ts, "angular rate")
    @test acceleration_by_name_ts.data == acceleration_ts.data
    @test angular_rate_by_name_ts.data == angular_rate_ts.data
    @test acceleration_by_name_ts.dimensions == acceleration_ts.dimensions
    @test angular_rate_by_name_ts.dimensions == angular_rate_ts.dimensions

    titled_dimension_ts = SystemsOfSystems.select(
        measurements_ts,
        "acceleration";
        title = "Acceleration",
    )
    @test titled_dimension_ts.title == "Acceleration"
    @test_throws "Could not find dimension missing. Valid labels: [\"acceleration\", \"angular rate\"]" SystemsOfSystems.select(measurements_ts, "missing")

    # Multiple labels produce tuples in the requested order.
    selected_dimensions_ts = SystemsOfSystems.select(
        measurements_ts,
        ["angular rate", "acceleration"],
    )
    @test selected_dimensions_ts.data == [
        (0.1, 1.0),
        (0.2, 2.0),
        (0.3, 3.0),
    ]
    @test selected_dimensions_ts.data isa Vector{Tuple{Float64, Float64}}
    @test selected_dimensions_ts.dimensions == [
        measurements_ts.dimensions[2],
        measurements_ts.dimensions[1],
    ]
    @test selected_dimensions_ts.title == measurements_ts.title
    @test selected_dimensions_ts.time == measurements_ts.time
    @test selected_dimensions_ts.path == measurements_ts.path
    @test selected_dimensions_ts.discrete == measurements_ts.discrete
    @test selected_dimensions_ts.groups == [
        "angular rate" => ["angular rate",],
        "acceleration" => ["acceleration",],
    ]
    @test selected_dimensions_ts.interpolator isa SystemsOfSystems.SampleAndHold

    # Tuples preserve the native type of each selected dimension.
    heterogeneous_ts = SystemsOfSystems.TimeSeries(;
        title = "Sensor Measurements",
        time = [0.0, 0.1],
        data = [
            HeterogeneousMeasurement(293.15, true),
            HeterogeneousMeasurement(294.15, false),
        ],
        time_dimension = SystemsOfSystems.Dimension("time", "s"),
        dimensions = ["temperature" => "K", "valid" => ""],
        path = "/sensor/measurement",
        discrete = true,
    )
    heterogeneous_dimensions_ts = SystemsOfSystems.select(
        heterogeneous_ts,
        ["valid", "temperature"],
    )
    @test heterogeneous_dimensions_ts.data == [
        (true, 293.15),
        (false, 294.15),
    ]
    @test heterogeneous_dimensions_ts.data isa Vector{Tuple{Bool, Float64}}

    # Selecting a dimension does not change the native payload stored by the source.
    @test measurements_ts[1] == (measurements_ts.time[1] => measurements[1])
    @test_throws BoundsError Dimensions.getdim(measurements_ts, 3)

end

@testset "VariableDescription metadata" begin

    offset_interpolator = OffsetLinearInterpolation(5.0)
    described_state = SystemsOfSystems.VariableDescription(
        0.0;
        title = "Described State",
        dimensions = [SystemsOfSystems.Dimension("state", ""),],
        groups = ["Custom State Axis" => ["state",],],
        interpolator = offset_interpolator,
    )
    default_described_state = SystemsOfSystems.VariableDescription(
        0.0;
        title = "Default Described State",
        dimensions = [SystemsOfSystems.Dimension("state", ""),],
        groups = [],
    )
    described_constant = SystemsOfSystems.VariableDescription(
        3.0;
        title = "Described Constant",
        dimensions = [SystemsOfSystems.Dimension("constant", "m"),],
        groups = ["Constant Axis" => ["constant",],],
        interpolator = offset_interpolator,
    )
    missing_constant = SystemsOfSystems.VariableDescription{Float64}(
        missing;
        title = "Unavailable Constant",
        dimensions = [SystemsOfSystems.Dimension("constant", "m"),],
    )

    @test described_state.interpolator === offset_interpolator
    @test described_state.groups == ["Custom State Axis" => ["state",],]
    @test ismissing(default_described_state.interpolator)
    @test isempty(default_described_state.groups)

    model_description = SystemsOfSystems.ModelDescription(;
        constants = (;
            raw_constant = 2.0,
            described_constant,
            raw_missing = missing,
            missing_constant,
        ),
        continuous_states = (;
            x = described_state,
            y = default_described_state,
        ),
        models = (;
            child = SystemsOfSystems.ModelDescription(;
                continuous_states = (;
                    z = 0.0,
                ),
            ),
            zebra = SystemsOfSystems.ModelDescription(),
            alpha = SystemsOfSystems.ModelDescription(),
        ),
    )

    basic_log, basic_history = Logs.create_log(
        Logs.BasicLogOptions(),
        model_description,
        SystemsOfSystems.Dimension("time", "s"),
    )
    @test basic_history.path == "/"
    @test basic_history.models.child.path == "/child"
    @test keys(basic_history.models) == (:child, :zebra, :alpha)
    @test basic_history.continuous_states.x.interpolator === offset_interpolator
    @test basic_history.continuous_states.y.interpolator isa SystemsOfSystems.LinearInterpolation
    @test basic_history.continuous_states.x.path == "/x"
    @test basic_history.models.child.continuous_states.z.path == "/child/z"
    @test basic_history.continuous_states.x.groups == described_state.groups
    @test isempty(basic_history.continuous_states.y.groups)
    @test basic_history.constants.raw_constant == 2.
    @test basic_history.constants.described_constant === described_constant
    @test ismissing(basic_history.constants.raw_missing)
    @test basic_history.constants.missing_constant === missing_constant
    Logs.close_log(basic_log)

    direct_filename = joinpath(out_dir, "variable_description_interpolator.h5")
    hdf5_log, hdf5_history = Logs.create_log(
        Logs.HDF5LogOptions(direct_filename),
        model_description,
        SystemsOfSystems.Dimension("time", "s"),
    )
    @test hdf5_history.continuous_states.x.interpolator === offset_interpolator
    @test hdf5_history.continuous_states.y.interpolator isa SystemsOfSystems.LinearInterpolation
    @test hdf5_history.continuous_states.x.path == "/x"
    @test hdf5_history.models.child.continuous_states.z.path == "/child/z"
    @test keys(hdf5_history.models) == (:child, :zebra, :alpha)
    @test hdf5_history.continuous_states.x.groups == described_state.groups
    @test isempty(hdf5_history.continuous_states.y.groups)
    @test hdf5_history.constants.raw_constant == 2.
    @test hdf5_history.constants.described_constant === described_constant
    @test ismissing(hdf5_history.constants.raw_missing)
    @test hdf5_history.constants.missing_constant === missing_constant
    Logs.close_log(hdf5_log)

    # Direct HDF5 logging and saving an in-memory log use separate writers. Both files must
    # restore custom interpolation and grouping rather than regenerating defaults.
    direct_log, direct_history = Logs.load_hdf5_log(direct_filename)
    @test direct_history.type === basic_history.type
    @test direct_history.continuous_states.x.interpolator isa OffsetLinearInterpolation
    @test direct_history.continuous_states.x.interpolator.offset == 5.
    @test direct_history.continuous_states.x.path == "/x"
    @test direct_history.models.child.continuous_states.z.path == "/child/z"
    @test keys(direct_history.models) == (:child, :zebra, :alpha)
    @test direct_history.continuous_states.y.interpolator isa
        SystemsOfSystems.LinearInterpolation
    @test direct_history.continuous_states.x.groups == described_state.groups
    @test isempty(direct_history.continuous_states.y.groups)
    @test direct_history.constants.raw_constant == 2.
    @test ismissing(direct_history.constants.raw_missing)
    @test typeof(direct_history.constants.missing_constant) == typeof(missing_constant)
    @test ismissing(direct_history.constants.missing_constant.value)
    direct_constant = direct_history.constants.described_constant
    @test typeof(direct_constant) == typeof(described_constant)
    @test direct_constant.value == described_constant.value
    @test direct_constant.title == described_constant.title
    @test direct_constant.dimensions == described_constant.dimensions
    @test direct_constant.groups == described_constant.groups
    @test direct_constant.interpolator isa OffsetLinearInterpolation
    @test direct_constant.interpolator.offset == 5.
    Logs.close_log(direct_log)

    saved_filename = joinpath(out_dir, "saved_variable_description_groups.h5")
    Logs.save_log_to_hdf5(saved_filename, basic_log)
    saved_log, saved_history = Logs.load_hdf5_log(saved_filename)
    @test saved_history.type === basic_history.type
    @test saved_history.continuous_states.x.interpolator isa OffsetLinearInterpolation
    @test saved_history.continuous_states.x.interpolator.offset == 5.
    @test saved_history.continuous_states.x.path == "/x"
    @test saved_history.models.child.continuous_states.z.path == "/child/z"
    @test keys(saved_history.models) == (:child, :zebra, :alpha)
    @test saved_history.continuous_states.y.interpolator isa
        SystemsOfSystems.LinearInterpolation
    @test saved_history.continuous_states.x.groups == described_state.groups
    @test isempty(saved_history.continuous_states.y.groups)
    @test saved_history.constants.raw_constant == 2.
    @test ismissing(saved_history.constants.raw_missing)
    @test typeof(saved_history.constants.missing_constant) == typeof(missing_constant)
    @test ismissing(saved_history.constants.missing_constant.value)
    saved_constant = saved_history.constants.described_constant
    @test typeof(saved_constant) == typeof(described_constant)
    @test saved_constant.value == described_constant.value
    @test saved_constant.title == described_constant.title
    @test saved_constant.dimensions == described_constant.dimensions
    @test saved_constant.groups == described_constant.groups
    @test saved_constant.interpolator isa OffsetLinearInterpolation
    @test saved_constant.interpolator.offset == 5.
    Logs.close_log(saved_log)

end

@testset "unsupported HDF5 constants" begin

    model_description = SystemsOfSystems.ModelDescription(;
        constants = (;
            unsupported = UnsupportedHDF5Constant(),
        ),
    )
    time_dimension = SystemsOfSystems.Dimension("time", "s")
    warning = r"/unsupported constant of type .*UnsupportedHDF5Constant"

    # Direct logging should keep the live initialization history usable while warning that
    # the constant will not be present when the file is loaded.
    direct_filename = joinpath(out_dir, "unsupported_direct_constant.h5")
    direct_log, direct_history = @test_logs (:warn, warning) Logs.create_log(
        Logs.HDF5LogOptions(direct_filename),
        model_description,
        time_dimension,
    )
    @test haskey(direct_history.constants, :unsupported)
    Logs.close_log(direct_log)
    loaded_direct_log, loaded_direct_history = Logs.load_hdf5_log(direct_filename)
    @test !haskey(loaded_direct_history.constants, :unsupported)
    Logs.close_log(loaded_direct_log)

    # Saving an in-memory log follows the same best-effort rule and diagnostic path.
    basic_log, = Logs.create_log(
        Logs.BasicLogOptions(),
        model_description,
        time_dimension,
    )
    saved_filename = joinpath(out_dir, "unsupported_saved_constant.h5")
    @test_logs (:warn, warning) Logs.save_log_to_hdf5(saved_filename, basic_log)
    loaded_saved_log, loaded_saved_history = Logs.load_hdf5_log(saved_filename)
    @test !haskey(loaded_saved_history.constants, :unsupported)
    Logs.close_log(loaded_saved_log)

end

end # TestTimeSeries
