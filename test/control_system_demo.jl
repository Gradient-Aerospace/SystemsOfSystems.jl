# We use this demo for documentation, testing, and benchmarking.
module ControlSystemDemo

export PlantSpecs, Plant, SensorSpecs, Sensor, PDControllerSpecs, PDController,
    ConstantTargetSpecs, ConstantTarget, ActuatorSpecs, Actuator,
    ClosedLoopSystemSpecs, ClosedLoopSystem

using Random: Xoshiro, randn
import Dimensions
import SystemsOfSystems
using SystemsOfSystems: ModelDescription, VariableDescription, RandomVariableDescription,
    RatesOutput, UpdatesOutput,
    ContinuousWhiteNoise, DiscreteWhiteNoise,
    RegularSchedule, is_triggering, on_triggering,
    Logs, SimOptions, Solvers, TimeSeries, Samplers, LoggingPolicies,
    simulate

#########
# Plant #
#########

# This models a second-order continuous-time system for position and velocity. Its input
# is an external force that drives the plant model.

# This is how the system is parameterized.
@kwdef struct PlantSpecs
    mass::Float64
    initial_position::Float64
    initial_velocity::Float64
    acceleration_noise_sigma::Float64
end

# This contains everything the model needs while running.
@kwdef struct Plant
    mass::Float64
    position::Float64
    velocity::Float64
    acceleration_noise::Float64
end

# This turns the specs into a description of the model. (It doesn't need the time or random-
# number generator inputs.)
function init(t, specs::PlantSpecs, seed)
    return ModelDescription(;
        type = Plant, # This is what tells it to build a Plant with this stuff.
        constants = (; # Constants we'll need while running
            mass = VariableDescription( # We can describe each variable in extra detail for plots and human output.
                specs.mass;
                title = "Mass",
                dimensions = ["mass" => "kg",],
            ),
        ),
        continuous_states = (;
            position = VariableDescription(
                specs.initial_position;
                title = "Plant Position",
                dimensions = ["position" => "m",],
            ),
            velocity = VariableDescription(
                specs.initial_velocity;
                title = "Plant Velocity",
                dimensions = ["velocity" => "m/s"],
            )
        ),
        continuous_random_variables = (;
            acceleration_noise = ContinuousWhiteNoise(specs.acceleration_noise_sigma),
        ),
        continuous_outputs = (;
            forces = VariableDescription(
                0.;
                title = "Total Forces on the Plant",
                dimensions = ["forces" => "N",],
            )
        ),
    )
end

# We'll make an accessor for the plant's position. (Maybe we'll have other plant models some
# day that don't have a position field but that still need to report their positions. This
# makes that easy.)
get_position(plant::Plant) = plant.position

# This is where we implement the model's dynamics -- functions that say how the model
# changes over time based on the inputs.
function rates(t, plant::Plant, actuator_force)
    forces = actuator_force
    acceleration = forces / plant.mass + plant.acceleration_noise
    return RatesOutput(;
        rates = (; # Derivatives of our continuous states
            position = plant.velocity,
            velocity = acceleration,
        ),
        outputs = (;
            forces = forces,
        ),
    )
end

##########
# Sensor #
##########

# This is a purely discrete model of the sensor. It measures the plant with a regular sample
# rate, adding noise to the plant's true position.

@kwdef struct SensorSpecs
    schedule::RegularSchedule # Defines sample period
    sigma_noise::Float64 # Standard deviation of noise to add to the measurement
    sigma_bias::Float64 # Standard deviation of the measurement bias
end

# The sensor's measurement is structured.
struct SensorMeasurement
    t::Float64 # Time at which measurement was made
    position::Float64 # The measured position (with noise and bias in it)
end

# This helps the plotting make sense of this structured type. It will automatically be able
# to plot SensorMeasurements over time thanks to this little function.
Dimensions.dimstyle(::Type{SensorMeasurement}) = Dimensions.StructDimensionStyle()

# Everything the sensor needs while running.
@kwdef struct Sensor
    schedule::RegularSchedule
    bias::Float64
    noise::Float64
    measurement::SensorMeasurement
end

# Describe all of the variables in the sensor model, with their initial conditions.
function init(t, specs::SensorSpecs, seed)

    # Draw the bias.
    bias = specs.sigma_bias * randn(Xoshiro(seed / "bias"))

    return ModelDescription(;
        type = Sensor,
        constants = (;
            bias = VariableDescription(
                bias;
                title = "Sensor Bias",
                dimensions = ["bias" => "m",],
            ),
        ),
        discrete_random_variables = (;
            noise = RandomVariableDescription{Float64}(
                DiscreteWhiteNoise(specs.sigma_noise);
                seed = seed / "noise",
                title = "Measurement White Noise",
                dimensions = ["noise" => "m",],
            ),
        ),
        discrete_states = (;
            measurement = VariableDescription(
                SensorMeasurement(0., 0.);
                title = "Sensor Measurement",
                dimensions = ["time" => "s", "position" => "m",],
            )
        ),
        schedules = (;
            # Tell the sim that this sensor needs a step on exactly its sample times.
            schedule = VariableDescription(
                specs.schedule;
                title = "Sensor Measurement Schedule",
                dimensions = ["period" => "s",],
            ),
        ),
    )

end

# Given the time and true position, this returns the most recent measurement. On off
# samples, it holds its last value.
function get_measurement(t, sensor::Sensor, true_position)

    # Our schedule triggers at t = 0, period, 2 * period, 3 * period, etc.
    if is_triggering(sensor.schedule, t)
        return SensorMeasurement(t, true_position + sensor.noise + sensor.bias)
    else
        return sensor.measurement
    end

end

# This says how the discrete states update on this sample.
function updates(t, sensor::Sensor, meas)

    # Only update when triggering.
    on_triggering(sensor.schedule, t) do
        return UpdatesOutput(;
            updates = (;
                measurement = meas, # Record our measurement (it's stateful).
            ),
        )
    end

end

###########
# Targets #
###########

# This model has no state or inputs. It simply outputs a target as a function of time. The
# layout of this model presents nothing new. It's discrete but doesn't have a regular sample
# period; it updates whenever any discrete step is taken.

@kwdef struct ConstantTargetSpecs
    constant_position::Float64
end

@kwdef struct ConstantTarget
    constant_position::Float64
end

function init(t, specs::ConstantTargetSpecs, seed)
    return ModelDescription(;
        type = ConstantTarget,
        constants = (;
            constant_position = VariableDescription(
                specs.constant_position;
                title = "Target Position",
                dimensions = ["target" => "m",],
            ),
        ),
        discrete_outputs = (;
            target = VariableDescription(
                specs.constant_position;
                title = "Target Position",
                dimensions = ["target" => "m",],
            ),
        ),
    )
end

function get_target_position(t, target::ConstantTarget)
    return target.constant_position
end

function updates(t, target::ConstantTarget, target_position)
    return UpdatesOutput(;
        outputs = (;
            target = target_position,
        ),
    )
end

################
# PDController #
################

# The controller is also nothing new as far as model layout goes. It's similar to the
# Sensor.

@kwdef struct PDControllerSpecs
    schedule::RegularSchedule
    p::Float64
    d::Float64
    initial_position::Float64
    initial_command::Float64
end

@kwdef struct PDController
    schedule::RegularSchedule
    p::Float64
    d::Float64
    position::Float64
    command::Float64
end

function init(t, specs::PDControllerSpecs, seed)
    return ModelDescription(;
        type = PDController,
        constants =  (;
            p = VariableDescription(
                specs.p;
                title = "Position Gain",
                dimensions = ["p" => "N / m",],
            ),
            d = VariableDescription(
                specs.d;
                title = "Velocity Gain",
                dimensions = ["d" => "N / (m/s)",],
            ),
        ),
        discrete_states = (;
            position = VariableDescription(
                specs.initial_position;
                title = "Position State",
                dimensions = ["position" => "m",],
            ),
            command = VariableDescription(
                specs.initial_command;
                title = "Command State",
                dimensions = ["command" => "N",],
            ),
        ),
        schedules = (;
            # Tell the sim that this controller acts on a regular period.
            schedule = VariableDescription(
                specs.schedule;
                title = "Controller Schedule",
                dimensions = ["period" => "s",],
            ),
        ),
    )
end

# Returns a fresh command or holds the last one.
function get_command(t, controller::PDController, target_position, meas)
    if is_triggering(controller.schedule, t)
        position_error = target_position - meas.position
        dt = controller.schedule.period
        velocity = (meas.position - controller.position) / dt
        command = controller.p * position_error - controller.d * velocity
        return command
    else
        return controller.command
    end
end

# Records what states should be updated. This holds the most recent command and position
# measurement.
function updates(t, controller::PDController, meas, command)
    on_triggering(controller.schedule, t) do
        return UpdatesOutput(;
            updates = (;
                command = command,
                position = meas.position,
            ),
        )
    end
end

############
# Actuator #
############

# This model takes in a target state and rises to it exponentially over time. It therefore
# has has both continuous and discrete parts.

@kwdef struct ActuatorSpecs
    time_constant::Float64
    initial_command::Float64
    initial_response::Float64
end

@kwdef struct Actuator
    time_constant::Float64
    command::Float64
    response::Float64
end

function init(t, specs::ActuatorSpecs, seed)
    return ModelDescription(;
        type = Actuator,
        constants = (;
            time_constant = VariableDescription(
                specs.time_constant;
                title = "First-Order Actuator Response Time Constant",
                dimensions = ["time_constant" => "s"]
            ),
        ),
        continuous_states = (;
            response = VariableDescription(
                specs.initial_response;
                title = "First-Order Actuator Response",
                dimensions = ["response" => ""],
            ),
        ),
        discrete_states = (;
            command = VariableDescription(
                specs.initial_command;
                title = "Actuator Command",
                dimensions = ["command" => ""],
            ),
        ),
    )
end

function get_actuator_response(t, actuator)
    return actuator.response
end

# For the continuous-time dynamics, this rises to the command.
function rates(t, actuator::Actuator)
    return RatesOutput(;
        rates = (;
            response = 1/actuator.time_constant * (actuator.command - actuator.response),
        ),
    )
end

# For the discrete-time dynamics, this records its command as state that it will use
# throughout its continuous-time dynamics.
function updates(t, actuator::Actuator, command)
    return UpdatesOutput(; # Updates discretely at any time (no need for a sample period).
        updates = (;
            command = command,
        ),
    )
end

####################
# ClosedLoopSystem #
####################

# This model's specs contain other models' specs. We can be specific about types but don't
# really need to be.
@kwdef struct ClosedLoopSystemSpecs
    plant       # Specs for the plant model
    sensor      # and for the sensor model
    actuator    # etc.
    target
    controller
end

# This model has no state or constants of its own; it just contains other models and
# connects them.
@kwdef struct ClosedLoopSystem
    plant::Plant
    sensor::Sensor
    actuator::Actuator
    target::ConstantTarget
    controller::PDController
end

# The ClosedLoopSystem model's initialization just initializes all of the sub-models with
# branched seeds. It also describes one top-level output signal we want.
function init(t, specs::ClosedLoopSystemSpecs, seed)

    # Initialize each submodel, as well as this model's own outputs.
    return ModelDescription(;
        type = ClosedLoopSystem,
        models = (;
            plant = init(t, specs.plant, seed / "plant"),
            sensor = init(t, specs.sensor, seed / "sensor"),
            target = init(t, specs.target, seed / "target"),
            controller = init(t, specs.controller, seed / "controller"),
            actuator = init(t, specs.actuator, seed / "actuator"),
        ),
        discrete_outputs = (;
            control_error = VariableDescription{Float64}(
                missing;
                title = "Control Error (Target - True Position)",
                dimensions = ["error" => "m",],
            ),
        ),
    )

end

# While this model has no continuous-time dynamics, it's responsible for routing things to
# its submodels so they can run theirs, and for forwarding what they return.
function rates(t, system::ClosedLoopSystem)

    # Calculate the things we'll need for the dynamics.
    actuator_force = get_actuator_response(t, system.actuator)

    # Run the continuous-time dynamics.
    return RatesOutput(;
        models = (;
            plant = rates(t, system.plant, actuator_force),
            actuator = rates(t, system.actuator),
        ),
    )

end

# This specifies how the discrete process unfolds and allows all submodels to say how they
# should update on this sample.
function updates(t, system::ClosedLoopSystem)

    # Here, we figure out everything that happens on this step, and then, below, we let each
    # model describe how that turns into its update.

    # First, measure the sensor.
    true_position = get_position(system.plant)
    meas = get_measurement(t, system.sensor, true_position)

    # Now figure out the command from the controller to the actuator.
    target_position = get_target_position(t, system.target)
    command = get_command(t, system.controller, target_position, meas)

    # This is the only model that knows both the target and the true position, so we'll
    # build the error signal here.
    control_error = target_position - true_position

    # Now that we have everything necessary to update the models, let them describe their
    # updates.
    return UpdatesOutput(;
        models = (;
            sensor = updates(t, system.sensor, meas),
            target = updates(t, system.target, target_position),
            controller = updates(t, system.controller, meas, command),
            actuator = updates(t, system.actuator, command),
        ),
        outputs = (;
            control_error,
        ),
    )

end

##############
# Simulation #
##############

"""
    default_closed_loop_system_specs()

Returns a useful default `ClosedLoopSystemSpecs`.
"""
function default_closed_loop_system_specs()
    return ClosedLoopSystemSpecs(
        plant = PlantSpecs(
            mass = 1.,
            initial_position = 0.,
            initial_velocity = 0.,
            acceleration_noise_sigma = 0.1,
        ),
        sensor = SensorSpecs(
            schedule = RegularSchedule(0.1),
            sigma_noise = 0.,
            sigma_bias = 0.,
        ),
        target = ConstantTargetSpecs(
            constant_position = 1.,
        ),
        controller = PDControllerSpecs(
            schedule = RegularSchedule(0.1),
            p = 8.,
            d = 4.,
            initial_position = 0.,
            initial_command = 0.,
        ),
        actuator = ActuatorSpecs(
            time_constant = 0.2,
            initial_command = 0.,
            initial_response = 0.,
        ),
    )
end

"""
    default_log()

Returns a sensible default log for the closed-loop system.

Here, the log's logging policy demonstrates hierarchical sampling: the root model allows
logging at 0.1-second intervals, and every descendant logs completely whenever traversal
reaches it.
"""
function default_log()
    return Logs.BasicLogOptions(;
        logging_policy = LoggingPolicies.RegexLoggingPolicy(
            [
                # Let the root control all logging.
                r"^/$" => LoggingPolicies.ModelLoggingPolicy(;
                    sampler = Samplers.RegularSampler(1//10),
                ),
                # For everything else, when the root allows logging, log everything.
                r"^/" => LoggingPolicies.ModelLoggingPolicy(;
                    sampler = Samplers.CompleteSampler(),
                ),
            ],
        ),
    )
end

"""
    simulate_closed_loop_system(; system_specs, log, solver)

Run the closed-loop control demo for 10s. Sensible defaults are included for `system_specs`,
`log`, and `solver`.

By default, the log configuration demonstrates hierarchical sampling: the root model allows
logging at 0.1-second intervals, and every descendant logs completely whenever traversal
reaches it.
"""
function simulate_closed_loop_system(;
    system_specs = default_closed_loop_system_specs(),
    log = default_log(),
    solver = Solvers.DormandPrince54Options(),
)
    return simulate(
        system_specs;
        init_fcn = ControlSystemDemo.init,
        rates_fcn = ControlSystemDemo.rates,
        updates_fcn = ControlSystemDemo.updates,
        t = (0, 10),
        options = SimOptions(; log, solver, time_dimension = "Time" => "s"),
    )
end

end
