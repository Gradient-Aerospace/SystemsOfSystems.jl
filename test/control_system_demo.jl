using Random: Xoshiro, randn, rand
import Dimensions
using SystemsOfSystems: ModelDescription, VariableDescription, is_regular_step_triggering, RatesOutput, UpdatesOutput

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
end

# This contains everything the model needs while running.
@kwdef struct Plant
    mass::Float64
    position::Float64
    velocity::Float64
end

# This turns the specs into a description of the model. (It doesn't need the time or random-
# number generator inputs.)
function init(t, specs::PlantSpecs, rng)
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
    acceleration = forces / plant.mass
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
    dt::Float64 # Sample period (s)
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
    dt::Float64
    bias::Float64
    noise::Float64
    measurement::SensorMeasurement
end

# Describe all of the variables in the plant model, with their initial conditions.
function init(t, specs::SensorSpecs, rng)
    return ModelDescription(;
        type = Sensor,
        constants = (;
            dt = VariableDescription(
                specs.dt;
                title = "Sensor Time Step",
                dimensions = ["dt" => "s",],
            ),
            bias = VariableDescription(
                specs.sigma_bias * randn(rng);
                title = "Sensor Bias",
                dimensions = ["bias" => "m",],
            ),
        ),
        discrete_random_variables = (;
            noise = VariableDescription(
                (t) -> specs.sigma_noise * randn(rng); # This builds a closure around whatever it needs.
                title = "Measurement White Noise",
                dimensions = ["noise" => "m",],
            ),
        ),
        discrete_states = (;
            measurement = VariableDescription(
                SensorMeasurement(0., 0.); # TODO: Make this missing?
                title = "Sensor Measurement",
                dimensions = ["time" => "s", "position" => "m",],
            )
        ),
        t_next = t + specs.dt, # Our next step comes from our regular sample period.
    )
end

# Given the time and true position, this returns the most recent measurement.
function get_measurement(t, sensor::Sensor, true_position)
    if is_regular_step_triggering(t, sensor.dt)
        return SensorMeasurement(t, true_position + sensor.noise + sensor.bias)
    else
        return sensor.measurement
    end
end

# This says how the discrete states update on this sample.
function updates(t, sensor::Sensor, meas)
    if is_regular_step_triggering(t, sensor.dt) # Will be true at 0, dt, 2dt, 3dt, etc.
        return UpdatesOutput(;
            updates = (;
                measurement = meas, # Record our measurement (it's stateful).
            ),
            t_next = t + sensor.dt, # Say when we need the solver to stop for us next.
        )
    else
        return UpdatesOutput() # There are no updates when it's not time to trigger.
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

function init(t, specs::ConstantTargetSpecs, rng)
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
    dt::Float64
    p::Float64
    d::Float64
    initial_position::Float64
    initial_command::Float64
end

@kwdef struct PDController
    dt::Float64
    p::Float64
    d::Float64
    position::Float64
    command::Float64
end

function init(t, specs::PDControllerSpecs, rng)
    return ModelDescription(;
        type = PDController,
        constants =  (;
            dt = VariableDescription(
                specs.dt;
                title = "Controller Time Step",
                dimensions = ["dt" => "s",],
            ),
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
        t_next = t + specs.dt, # Don't forget to say when we trigger next!
    )
end

function get_command(t, controller::PDController, target_position, meas)
    if is_regular_step_triggering(t, controller.dt)
        position_error = target_position - meas.position
        velocity = (meas.position - controller.position) / controller.dt
        command = controller.p * position_error - controller.d * velocity
        return command
    else
        return controller.command # or nothing
    end
end

function updates(t, controller::PDController, meas, command)
    if is_regular_step_triggering(t, controller.dt)
        return UpdatesOutput(;
            updates = (;
                command = command,
                position = meas.position,
            ),
            t_next = t + controller.dt,
        )
    else
        return UpdatesOutput()
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

function init(t, specs::ActuatorSpecs, rng)
    return ModelDescription(;
        type = Actuator,
        constants = (;
            time_constant = VariableDescription(
                0.1;
                title = "First-Order Actuator Response Time Constant",
                dimensions = ["tau" => "s"]
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

# This function creates a new random number generator based on a "salt" (some random draw)
# and a string. This is a useful modeling paradigm allowing a parent model to make new
# RNGs for its sub-models. If they share a randomly-generated "salt" but have unique names,
# then the result is that (1) changing the top level seed will change all draws everywhere,
# but (2) changes to any one model don't affect any other models' RNG streams. You don't
# have to do this, but it's an excellent pattern for modeling random variables inside of
# systems of systems. See how we use it, below.
child_rng(salt, name) = Xoshiro(salt + hash(name))

# The ClosedLoopSystem model's initialization just initializes all of the sub-models and
# gives each one a unique random number generator. It also describes one top-level output
# signal we want.
function init(t, specs::ClosedLoopSystemSpecs, rng)

    # Make a random draw we'll share in generating the sub-model RNGs.
    salt = rand(rng, Int64)

    # Initialize each submodel, as well as this model's own outputs.
    return ModelDescription(;
        type = ClosedLoopSystem,
        models = (;
            plant = init(t, specs.plant, child_rng(salt, "plant")),
            sensor = init(t, specs.sensor, child_rng(salt, "sensor")),
            target = init(t, specs.target, child_rng(salt, "target")),
            controller = init(t, specs.controller, child_rng(salt, "controller")),
            actuator = init(t, specs.actuator, child_rng(salt, "actuator")),
        ),
        discrete_outputs = (;
            control_error = VariableDescription{Union{Missing, Float64}}(
                0.; # TODO: Make this missing?
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

@testset failfast=false "control system demo with $solver_type solver, $log_type logs" for solver_type in ["rk4", "dp54"], log_type in ["ram", "hdf5"]

    dt_rk4 = 0.06 # Deliberately chosen to be inconsistent with the discrete systems' sample rates
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

    # Set the parameters for all of the models.
    system_specs = ClosedLoopSystemSpecs(
        plant = PlantSpecs(
            mass = 1.,
            initial_position = 0.,
            initial_velocity = 0.,
        ),
        sensor = SensorSpecs(
            dt = 0.1,
            sigma_noise = 0.,
            sigma_bias = 0.,
        ),
        target = ConstantTargetSpecs(
            constant_position = 1.,
        ),
        controller = PDControllerSpecs(
            dt = 0.1,
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

    # Run the sim.
    history, t, system = simulate(
        system_specs;
        init_fcn = init,
        rates_fcn = rates,
        updates_fcn = updates,
        t = (0, 10),
        options = SimOptions(;
            solver,
            log,
            time_dimension = "Time" => "s",
        ),
    )

    @test t == 10
    @test history["/sensor"]["measurement"].time == collect(0. : system_specs.sensor.dt : t)
    @test history["/sensor"]["measurement"].data[end].t == t

    Logs.close_log(history.log)

    # Also, test for type stability. First, get the pieces we'll need.
    md = init(0//1, system_specs, Xoshiro(1))
    ommd = SystemsOfSystems.strip_fluff_from_model_description(md)
    msd = SystemsOfSystems.draw_wd(0//1, ommd, ommd)

    # See that we can convert the model description to a model with a known type.
    @inferred SystemsOfSystems.model(msd)

    # Our rates function should be type stable.
    @inferred rates(t, system)

    # Our `updataes` is not type stable; on "off" samples, the UpdatesOutput will be empty,
    # which is a different type.
    # @inferred updates(t, system)
    empty_updates_output = UpdatesOutput()
    command = 0.
    meas = get_measurement(t, system.sensor, 0.)
    @inferred typeof(empty_updates_output) updates(t, system.sensor, meas)
    @inferred typeof(empty_updates_output) updates(t, system.controller, meas, command)
    @inferred updates(t, system.actuator, command) # We can test this one though.

    # TODO: Now test with the internal solve/update functions.

end
