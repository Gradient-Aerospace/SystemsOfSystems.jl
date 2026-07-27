# Control System Example

In this example, we're going to develop a closed-loop control system that features:

* A "plant", the system whose dynamics we want to control
* A sensor that make noisy measurements of the plant
* An actuator that provides an input to the plant
* A target generator that produces the target state we want to drive the plant to
* And a control system the uses the sensor and sends commands to the actuator to drive the plant to the target state.

```
+-- Closed-Loop System ---------------------------------------------------+
|                                                                         |
|  +-------------+                                                        |
|  |   Target    |                                                        |
|  |  Generator  |                                                        |
|  +-----+-------+                                                        |
|        |                                                                |
|        | Target                                                         |
|        v                                                                |
|  +-----+-----+              +------------+            +--------------+  |
|  |  Control  |---Command--->|  Actuator  |---Input--->|    Plant     |  |
|  |  System   |              +------------+            |  (Dynamics)  |  |
|  +-----+-----+                                        +------+-------+  |
|        ^                                                     |          |
|        |                          +-----------+              |          |
|        |                          |  Sensor   |              |          |
|        +---------Measurement------|  (Noisy)  |<-------------+          |
|                                   +-----------+                         |
|                                                                         |
+-------------------------------------------------------------------------+
```

To do this, we'll see several different ways to model using SystemsOfSystems, including:

* Continuous systems
* Discrete systems
* Random variables
* Schedules
* Models that contain sub-models

This is a working example. The the outputs and plots and generated as part of generating this documentation. One can copy all of this into a script and product the exact same results.

(Quick note: Nothing about SystemsOfSystems is specific to control theory or closed-loop systems. It's simply a common simulation pattern that we can implement with general simulation engine like SystemsOfSystems.)

## Imports

Since this is a working example, we'll import everything we need up front:

```@example controls
import Random
import SystemsOfSystems
import Dimensions # We'll want this to label our sensor measurement
import GLMakie # The plotting package we'll use
```

## Plant

We'll start with a very simple plant. The dynamics as as follows:

```math
\ddot{x} = \frac{1}{m} \left( -x + f_a + \nu \right)
```

where ``x`` is the plant state (position), ``m`` is the mass, ``f_a`` is the force from the actuator, and ``\nu`` is white noise.

### Plant Implementation

First, we want a way to parameterize this system, to set the constants and initial conditions. Let's make a quick type for that.

```@example controls
# This is how the system is parameterized.
@kwdef struct PlantSpecs
    mass::Float64
    initial_position::Float64
    initial_velocity::Float64
    sigma_noise::Float64
end
;
```

(In case you're new to Julia, the `@kwdef` here says, "Make me a constructor that has keyword arguments for this, so we can call `PlantSpecs(; mass = 1., initial_position = 2., ...)`, etc.)

At any given moment while the plant is running in the sim, we want to be able to access its mass, position, velocity, and noise input, so let's make the "model" form:

```@example controls
# This contains everything the model needs while running.
@kwdef struct Plant
    mass::Float64
    position::Float64
    velocity::Float64
    noise::Float64
end
;
```

We need a function that tells the sim what constants our model should have, what states, the random variables, and also any other outputs we want logged. The input to this function will be time, our `PlantSpecs from above, and a seed that we can use for random initial conditions.

```@example controls
# This turns the specs into a description of the model.
function init(t, specs::PlantSpecs, seed)
    return SystemsOfSystems.ModelDescription(;
        # This is what tells the sim to build a Plant with this stuff.
        type = Plant,
        # Constants we'll need while running
        constants = (;
            # We can describe each variable in detail for plots and human output.
            mass = SystemsOfSystems.VariableDescription(
                specs.mass;
                title = "Mass",
                dimensions = ["mass" => "kg",],
            ),
        ),
        # We treat position and velocity as two state variables, since this is
        # a 2nd-order system.
        continuous_states = (;
            position = SystemsOfSystems.VariableDescription(
                specs.initial_position;
                title = "Plant Position",
                dimensions = ["position" => "m",],
            ),
            velocity = SystemsOfSystems.VariableDescription(
                specs.initial_velocity;
                title = "Plant Velocity",
                dimensions = ["velocity" => "m/s"],
            )
        ),
        # We can ask for a continuous white noise source. This will all be handled
        # properly by the integrator.
        continuous_random_variables = (;
            noise = SystemsOfSystems.RandomVariableDescription{Float64}(
                SystemsOfSystems.ContinuousWhiteNoise(specs.sigma_noise);
                title = "Continuous White Noise Force Input",
                dimensions = ["noise" => "N",],
                seed = seed / "noise",
            ),
        ),
        # Further, we have pieces that aren't part of the Plant but that are things
        # we want logged as byproducts of our calculations.
        continuous_outputs = (;
            forces = SystemsOfSystems.VariableDescription(
                0.;
                title = "Total Forces on the Plant",
                dimensions = ["forces" => "N",],
            )
        ),
    )
end
;
```

The describes all of the variables that the plant cares about. (We don't _need_ to use `VariableDescriptions` here; we could just use raw numbers/structs, but the descriptions show up nicely in plots, so we'll use them in this example.)

TODO: branching

The plant has only continuous-time dynamics. Let's write a function that takes in the current time, plant model, and actuator force and returns the derivatives of the state variables, plus the extra output we said we wanted logged. Of

```@example controls
# This is where we implement the model's dynamics -- functions that say how the model
# changes over time based on the inputs.
function rates(t, plant::Plant, actuator_force)
    forces = -plant.position + actuator_force + plant.noise
    acceleration = forces / plant.mass
    return SystemsOfSystems.RatesOutput(;
        rates = (; # Derivatives of our continuous states
            position = plant.velocity,
            velocity = acceleration,
        ),
        outputs = (;
            forces = forces,
        ),
    )
end
;
```

Note: We called our functions `init` and `rates`, but the names are arbitrary. We can choose whatever names we like.

Finally, other models will need the plants position (the sensor will, in order to take a measurement of it), so let's provide a function that returns the position so that the sensor model doesn't have to access the plant's state directly.

```@example controls
get_position(plant::Plant) = plant.position
```

### Plant Simulation

Before we move on with our example, we can pause here and simulate the plant by itself, to make sure things are working. We'll need to do one extra thing: wrap our `rates` function with a function that supplies the non-existant actuator force (0 for this example).

```@example controls
plant_specs = PlantSpecs(;
    mass = 1.,
    initial_position = 0.,
    initial_velocity = 0.,
    sigma_noise = 1.,
)
history, t, plant = SystemsOfSystems.simulate(
    plant_specs;
    t = (0, 10),
    init_fcn = init,
    rates_fcn = (t, plant) -> rates(t, plant, 0.),
)

history
```

Let's take a look at the time history of the root model (the plant, of course):

```@example controls
history["/"]
```

We can access the position itself as:

```@example controls
history["/"]["position"]
```

And, while we're here, let's plot it.

```@example controls
SystemsOfSystems.plot_ts(history["/"]["position"])
```

In fact, let's run 15 simulations, returning the `position` time series from each and then plot them all. Here, we'll write a loop that generates an array of 15 time histories of position.

```@example controls
# Make an array of position histories, one for each seed.
position_histories = map(1:15) do seed

    # Run the sim for this see.
    history, t, plant = SystemsOfSystems.simulate(
        plant_specs;
        t = (0, 10),
        init_fcn = init,
        rates_fcn = (t, plant) -> rates(t, plant, 0.),
        seed = seed,
    )

    # Pull out the position history. Also, give it a label for
    # the plot.
    return "Run $seed" => history["/"]["position"]

end

# Plot all of those as separate lines with the given labels.
SystemsOfSystems.plot_ts(position_histories)
```

We can see the sinusoidal motion we'd expect to result for a system like this, driven by noise. At this point, we can have some faith that our plant model is doing what we intend. Let's continue with the remaining systems.

## Sensor

The sensor will be a discrete system, with discrete state and discrete noise. It will also be triggered at a regular rate. Here's how we can implement that.

We start with the specifications (options, parameters, whatever).

```@example controls
@kwdef struct SensorSpecs
    schedule::SystemsOfSystems.RegularSchedule # Defines the sample period
    sigma_noise::Float64 # Standard deviation of noise to add to the measurement
    sigma_bias::Float64 # Standard deviation of the measurement bias
end
```

The sensor will product a specific, structured type, so let's implement that type.

```@example controls
struct SensorMeasurement
    t::Float64 # Time at which measurement was made
    position::Float64 # The measured position (with noise and bias in it)
end
```

The measurement itself will be stateful. Once the measurement is generated, it will be help constant until the next measurement is made. Hence, the measurement is part of the `Sensor` model:

```@example controls
# Everything the sensor needs while running.
@kwdef struct Sensor
    schedule::SystemsOfSystems.RegularSchedule
    bias::Float64 # Constant
    noise::Float64 # A random input on each step
    measurement::SensorMeasurement # The most recent measurement
end
```

Let's put together the initialization function and model description.

```@example controls
# Describe all of the variables in the sensor model, with their initial conditions.
function init(t, specs::SensorSpecs, seed)

    # Draw the bias. Here, we "branch" the seed for the bias. See below.
    rng = Random.Xoshiro(seed / "bias")
    bias = specs.sigma_bias * Random.randn(rng)

    return SystemsOfSystems.ModelDescription(;
        type = Sensor,
        constants = (;
            bias = SystemsOfSystems.VariableDescription(
                bias;
                title = "Sensor Bias",
                dimensions = ["bias" => "m",],
            ),
        ),
        discrete_random_variables = (;
            noise = SystemsOfSystems.RandomVariableDescription{Float64}(
                SystemsOfSystems.DiscreteWhiteNoise(specs.sigma_noise);
                title = "Measurement Noise",
                dimensions = ["noise" => "m",],
                seed = seed / "noise",
            ),
        ),
        discrete_states = (;
            measurement = SystemsOfSystems.VariableDescription(
                SensorMeasurement(0., 0.);
                title = "Sensor Measurement",
                dimensions = ["time" => "s", "position" => "m",],
            )
        ),
        schedules = (;
            # Tell the sim that this sensor needs a step on exactly its sample times.
            schedule = SystemsOfSystems.VariableDescription(
                specs.schedule;
                title = "Sensor Measurement Schedule",
                dimensions = ["period" => "s",],
            ),
        ),
    )

end
```

Since we're using the `SensorMeasurement` struct as state, and since we want plots of the states, we need to tell SystemsOfSystems how to break this down into the different dimensions (the different lines in the plot). SystemsOfSystems will look to the Dimensions package for this behavior, so we use the Dimension method for saying, "The `SensorMeasurement` has two dimension, and those dimensions are its two fields -- a very normal interpretation of "dimensions" for a struct."

```@example controls
# This helps the plotting make sense of this structured type. It will automatically be able
# to plot SensorMeasurements over time thanks to this little function.
Dimensions.dimstyle(::Type{SensorMeasurement}) = Dimensions.StructDimensionStyle()
```

Let's provide a function so that other models can ask the sensor for its measurement at a given time.

```@example controls
# Given the time and true position, this returns the most recent measurement.
# When it's time for a new measurement, it will make one. Otherwise, it will
# return the prior measurement (its `measurement` state).
function get_measurement(t, sensor::Sensor, true_position)

    # Our schedule triggers at t = 0, period, 2 * period, 3 * period, etc.
    if SystemsOfSystems.is_triggering(sensor.schedule, t)
        return SensorMeasurement(t, true_position + sensor.noise + sensor.bias)
    else
        return sensor.measurement
    end

end
```

Now we specify the discrete dynamics. This is very simple. When triggering, we record the new measurement. (Otherwise, we do nothing.) The measurement will be an _input_ to this function. It will be generated with the above function. We'll see how this all comes together in the closed-loop top model that routes everything together.

```@example controls
# This says how the discrete states update on this sample.
function updates(t, sensor::Sensor, meas)

    # Only update when triggering.
    SystemsOfSystems.on_triggering(sensor.schedule, t) do
        return SystemsOfSystems.UpdatesOutput(;
            updates = (;
                measurement = meas, # Record our measurement (it's stateful).
            ),
        )
    end

end
```

## Actuator

On the other side of the plant is the actuator. That model is both continuous and discrete. Its discrete state is the incoming command from the controller. Its continuous state is its reponse to that command. We'll allow this to be a simple, first-order system.

The pattern of writing a model's specifications and "model form" should be familiar by now. There's nothing new here, so we'll write a bit more all at once:

```@example controls
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
    return SystemsOfSystems.ModelDescription(;
        type = Actuator,
        constants = (;
            time_constant = SystemsOfSystems.VariableDescription(
                specs.time_constant;
                title = "First-Order Actuator Response Time Constant",
                dimensions = ["time_constant" => "s"]
            ),
        ),
        continuous_states = (;
            response = SystemsOfSystems.VariableDescription(
                specs.initial_response;
                title = "First-Order Actuator Response",
                dimensions = ["response" => ""],
            ),
        ),
        discrete_states = (;
            command = SystemsOfSystems.VariableDescription(
                specs.initial_command;
                title = "Actuator Command",
                dimensions = ["command" => ""],
            ),
        ),
    )
end

# For the continuous-time dynamics, the response rises to the command.
function rates(t, actuator::Actuator)
    return SystemsOfSystems.RatesOutput(;
        rates = (;
            response = 1/actuator.time_constant * (actuator.command - actuator.response),
        ),
    )
end

# For the discrete-time dynamics, this records its command as state that it will use
# throughout its continuous-time dynamics. (We don't use a sample period here. We assume
# this updates on _any_ sample.)
function updates(t, actuator::Actuator, command)
    return SystemsOfSystems.UpdatesOutput(;
        updates = (;
            command = command,
        ),
    )
end

# An accessor that outside models can use:
get_actuator_response(t, actuator) = actuator.response
```

## Target

Before we write the controller, let's make something that generates the target that the controller will track.

We'll start with a simple target generator: a constant target.

```@example controls
@kwdef struct ConstantTargetSpecs
    constant_position::Float64
end

@kwdef struct ConstantTarget
    constant_position::Float64
end

# Since the target for this model is stateless, let's log it as an output.
function init(t, specs::ConstantTargetSpecs, seed)
    return SystemsOfSystems.ModelDescription(;
        type = ConstantTarget,
        constants = (;
            constant_position = SystemsOfSystems.VariableDescription(
                specs.constant_position;
                title = "Target Position",
                dimensions = ["target" => "m",],
            ),
        ),
        discrete_outputs = (;
            target = SystemsOfSystems.VariableDescription(
                specs.constant_position;
                title = "Target Position",
                dimensions = ["target" => "m",],
            ),
        ),
    )
end

function updates(t, target::ConstantTarget, target_position)
    return SystemsOfSystems.UpdatesOutput(;
        outputs = (;
            target = target_position,
        ),
    )
end

get_target_position(t, target::ConstantTarget) = target.constant_position
```

## Controller

Now we can integrate all of these systems with the controller. We'll use a typical PID for this.

```@example controls
@kwdef struct PIDControllerSpecs
    schedule::SystemsOfSystems.RegularSchedule
    k_p::Float64 # Proportional gain
    k_i::Float64 # Integral gain
    k_d::Float64 # Derivative gain
    initial_position::Float64 # Initial value of position state
    initial_command::Float64 # Initial command output
    initial_integral::Float64
end

@kwdef struct PIDController
    schedule::SystemsOfSystems.RegularSchedule
    k_p::Float64 # Proportional gain
    k_i::Float64 # Integral gain
    k_d::Float64 # Derivative gain
    position::Float64 # Last position measurement
    command::Float64 # Last command output
    integral::Float64
end

function init(t, specs::PIDControllerSpecs, seed)
    return SystemsOfSystems.ModelDescription(;
        type = PIDController,
        constants =  (;
            k_p = SystemsOfSystems.VariableDescription(
                specs.k_p;
                title = "Position Gain",
                dimensions = ["p" => "N / m",],
            ),
            k_i = SystemsOfSystems.VariableDescription(
                specs.k_i;
                title = "Integral Gain",
                dimensions = ["i" => "N / (m s)",],
            ),
            k_d = SystemsOfSystems.VariableDescription(
                specs.k_d;
                title = "Velocity Gain",
                dimensions = ["d" => "N / (m/s)",],
            ),
        ),
        discrete_states = (;
            position = SystemsOfSystems.VariableDescription(
                specs.initial_position;
                title = "Position State",
                dimensions = ["position" => "m",],
            ),
            command = SystemsOfSystems.VariableDescription(
                specs.initial_command;
                title = "Command State",
                dimensions = ["command" => "N",],
            ),
            integral = SystemsOfSystems.VariableDescription(
                specs.initial_integral;
                title = "Integral State",
                dimensions = ["integral" => "m s",],
            ),
        ),
        schedules = (;
            # Tell the sim that this controller acts on a regular period.
            schedule = SystemsOfSystems.VariableDescription(
                specs.schedule;
                title = "Controller Schedule",
                dimensions = ["period" => "s",],
            ),
        ),
    )
end

# Returns a fresh command or holds the last one.
function get_command(t, controller::PIDController, target_position, meas)

    if SystemsOfSystems.is_triggering(controller.schedule, t)

        # Divide the difference between the current and last measured
        # position to get an approximation of velocity.
        dt = controller.schedule.period
        velocity = (meas.position - controller.position) / dt

        # Now make the command from the three different parts.
        position_error = target_position - meas.position
        command = (
            controller.k_p * position_error
            + controller.k_i * controller.integral
            - controller.k_d * velocity
        )
        return command

    else

        return controller.command

    end

end

# Records what states should be updated. This holds the most recent command and position
# measurement.
function updates(t, controller::PIDController, target, measurement, command)

    SystemsOfSystems.on_triggering(controller.schedule, t) do

        # Update the integral.
        dt = controller.schedule.period
        integral = controller.integral + (target - measurement.position) * dt

        # Record the command, latest measurement, and updated integral.
        return SystemsOfSystems.UpdatesOutput(;
            updates = (;
                command = command,
                position = measurement.position,
                integral = integral,
            ),
        )

    end

end
```

## Closed-Loop System

Finally, we're ready to bring all of those systems together, essentially routing from one to the other. The closed-loop system will be the top-level system in our simulation (will have no inputs from outside or outputs going to anything else). It will also have no state. Its only job is to contain/route the sub-models.

The specifications for a closed-loop system are simply the set of specifications for the sub-models.

```@example controls
@kwdef struct ClosedLoopSystemSpecs
    plant       # Specs for the plant model
    sensor      # and for the sensor model
    actuator    # etc.
    target
    controller
end
```

We don't specify types for the above because there's no point, and because we might want to swap out one model type for another, compatible model type.

Similar, the "model form" itself just holds the other model forms.

```@example controls
@kwdef struct ClosedLoopSystem
    plant
    sensor
    actuator
    target
    controller
end
```

When we initialize this model, that will just consist of initializing the sub-models. We'll also declare one output variable: the control error.

```@example controls
function init(t, specs::ClosedLoopSystemSpecs, seed)

    # Initialize each submodel, as well as this model's own outputs.
    return SystemsOfSystems.ModelDescription(;
        type = ClosedLoopSystem,
        models = (;
            plant = init(t, specs.plant, seed / "plant"),
            sensor = init(t, specs.sensor, seed / "sensor"),
            target = init(t, specs.target, seed / "target"),
            controller = init(t, specs.controller, seed / "controller"),
            actuator = init(t, specs.actuator, seed / "actuator"),
        ),
        discrete_outputs = (;
            control_error = SystemsOfSystems.VariableDescription{Float64}(
                missing;
                title = "Control Error (Target - True Position)",
                dimensions = ["error" => "m",],
            ),
        ),
    )

end
```

There is no `control_error` on the initial sample; hence, we have a `missing` there, and we declare that this will, when its available, be a `Float64`.

Note that we "branch" the seed here for each sub-model. This means that, if the plant and sensor both take draws, they won't be taking the same draws, and their draws won't interact with each other. They will have totally separate (but predictable) streams.

Just like the `init` function, the job of this model's `rates` function is to gather up the `RatesOutputs` for its sub-models. The plant's `rates` function depends on the actuator force input, so we get that from the actuator. Now, finally, we start to see where we use those accessors we made.

```@example controls
function rates(t, system::ClosedLoopSystem)

    # Calculate the things we'll need for the dynamics.
    actuator_force = get_actuator_response(t, system.actuator)

    # Run the continuous-time dynamics.
    return SystemsOfSystems.RatesOutput(;
        models = (;
            plant = rates(t, system.plant, actuator_force),
            actuator = rates(t, system.actuator),
        ),
    )

end
```

The `updates` function involves more routing, using our remaining accessors.

```@example controls
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
    return SystemsOfSystems.UpdatesOutput(;
        models = (;
            sensor = updates(t, system.sensor, meas),
            target = updates(t, system.target, target_position),
            controller = updates(t, system.controller, target_position, meas, command),
            actuator = updates(t, system.actuator, command),
        ),
        outputs = (;
            control_error,
        ),
    )

end
```

## Simulation

Now we're ready to simulate the complete system.

```@example controls
system_specs = ClosedLoopSystemSpecs(
    plant = PlantSpecs(
        mass = 1.,
        initial_position = 0.,
        initial_velocity = 0.,
        sigma_noise = 0.1,
    ),
    sensor = SensorSpecs(
        schedule = SystemsOfSystems.RegularSchedule(0.05),
        sigma_noise = 0.01,
        sigma_bias = 0.1,
    ),
    target = ConstantTargetSpecs(
        constant_position = 1.,
    ),
    controller = PIDControllerSpecs(
        schedule = SystemsOfSystems.RegularSchedule(0.05),
        k_p = 11.,
        k_i = 8.,
        k_d = 6.,
        initial_position = 0.,
        initial_command = 0.,
        initial_integral = 0.,
    ),
    actuator = ActuatorSpecs(
        time_constant = 0.04,
        initial_command = 0.,
        initial_response = 0.,
    ),
)

history, final_time, final_system = SystemsOfSystems.simulate(
    system_specs;
    t = (0, 10),
    init_fcn = init,
    rates_fcn = rates,
    updates_fcn = updates,
)

history
```

Let's see if we're driving down the control error.

```@example controls
SystemsOfSystems.plot_ts(history["/"]["control_error"])
```

Note that this is the true control error, the target position minus the truth position. There's a steady-state offset because our sensor has a bias.

Let's make a more involved plot. Let's plot the target position, measured position, and the truth all together, and let's put the integral term beneath it.

```@example controls
SystemsOfSystems.plot_ts(
    [
        [
            "plant" => history["/plant"]["position"],
            "measured" => history["/controller"]["position"],
            "target" => history["/target"]["target"],
        ],
        history["/controller"]["integral"]
    ]
)
```

Clearly, the controller is driving the _measurement_ to the target; that's all this simple controller can really do in this situation.

The performance certainly appears to be correct.

Note the difference between lines for continuous-time variables and points for discrete variables.

Let's take a look at `final_system`, the complete model form at the end of the simulation:

```@example controls
dump(final_system)
```
