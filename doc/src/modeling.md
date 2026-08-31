# Modeling

A SystemsOfSystems model is described by three functions:

* An initialization function that defines the model's structure and initial values
* A continuous-time dynamics function that calculates derivatives and continuous outputs
* A discrete-time dynamics function that applies state changes and calculates discrete outputs

The functions can have any names. They are supplied to [`simulate`](@ref) as `init_fcn`, `rates_fcn`, and `updates_fcn`.

SystemsOfSystems does not mutate the model passed to these functions. Instead, it constructs a fresh model from its constants, states, random variables, schedules, resources, and submodels whenever the simulation state changes. Model functions calculate and return their results without mutating the model or performing hidden state changes.

The [Control System Example](@ref) develops a complete hierarchical model. This page describes the individual pieces that can be used to build one.

## Primary Function Outputs

### Initialization

The initialization function is called as `init_fcn(t_start, user_data, seed)`. It returns a [`ModelDescription`](@ref), which defines the complete and fixed structure of the model.

```julia
function init(t, specs, seed)
    return ModelDescription(;
        type = MyModel,
        constants = (;
            mass = specs.mass,
        ),
        continuous_states = (;
            position = specs.initial_position,
            velocity = specs.initial_velocity,
        ),
        discrete_states = (;
            mode = :nominal,
        ),
        models = (;
            sensor = sensor_init(t, specs.sensor, seed / "sensor"),
        ),
    )
end
```

The purpose of a `ModelDescription` is to describe each "variable" in the model, where a variable can be a constant, state, output, random variable, submodel, resource, or schedule. Each variable name must be unique within its model.

The model will be constructed by calling the given `type` with each variable as a keyword argument. If no `type` is given, the model will be a named tuple of all of the variables.

Here is an example "model form" with all of the above variables in it. (Note that `@kwdef` adds a keyword constructor for the struct.)

```julia
@kwdef struct MyModel
    mass
    position
    velocity
    mode
    sensor
end
```

There is no fixed limit on the number of submodels. However, the model hierarchy is encoded in concrete named-tuple types so that simulation can be fast. A very wide model with a large number of direct submodels will increase compilation time and compiler memory use. Large systems can be grouped into meaningful intermediate models instead of placing every leaf model directly under the root.

Raw values are sufficient for constants, states, and outputs. A [`VariableDescription`](@ref) adds a plot title, dimensions, units, dimension groups, and an optional time-series interpolation policy.

```@docs
SystemsOfSystems.ModelDescription
```

Note that the top-level seed input to `init_fcn` is a `BranchingSeed`.

```@docs
SystemsOfSystems.BranchingSeed
SystemsOfSystems.branch
```

The following sections describe the types of variables in more depth. Except where noted, variables can be decorated with a `VariableDescription`:

```@docs
SystemsOfSystems.VariableDescription
```

#### Random Variables

Random variables are declared separately from states. SystemsOfSystems owns their random number generators and supplies fresh draws to the model. This keeps random processes repeatable from the top-level `seed` passed to [`simulate`](@ref), and it allows the sim to handle random draws properly in coordination with the solver.

A random variable can be any callable (function or functor), or it can be a [`RandomVariableDescription`](@ref), which further specifies the value type, seed, title, dimensions, units, and plotting groups.

The initialization seed can be branched for each logically independent random process:

```julia
continuous_random_variables = (;
    force_noise = RandomVariableDescription{Float64}(
        ContinuousWhiteNoise(0.1);
        seed = seed / "force_noise",
        title = "Force Noise",
        dimensions = ["force" => "N"],
    ),
)
```

Branching makes one process independent of the number of draws taken by another process. Changing the top-level simulation seed still changes every branch predictably.

```@docs
SystemsOfSystems.RandomVariableDescription
```

##### Continuous Random Variables

A continuous random variable can be any callable (e.g., function) that accepts `(rng, t_km1, dt_f)`, where `t_km1` is the exact start time and `dt_f` is the floating-point duration of the proposed interval. SystemsOfSystems draws it for that interval and makes the result available as a field of the model during rate calculations.

[`ContinuousWhiteNoise`](@ref) is the built-in Gaussian white-noise process. Its `sigma` scales a draw that is divided by the square root of the interval, so its integrated effect has the expected continuous-time scaling.

```@docs
SystemsOfSystems.ContinuousWhiteNoise
```

##### Discrete Random Variables

A discrete random variable can be any callable type that accepts `(rng, t)`. SystemsOfSystems draws it at every simulation time before calling `updates_fcn`. The latest draw is available as a field of the model.

[`DiscreteWhiteNoise`](@ref) is the built-in Gaussian discrete white-noise process. Its `sigma` is the standard deviation of each draw.

```@docs
SystemsOfSystems.DiscreteWhiteNoise
```

#### Schedules

A schedule declares times at which the simulation must take a step (call `updates_fcn`). Schedules can be declared as variables in the `schedules` field of a `ModelDescription`; each schedule is then available as a field of the running model.

`RegularSchedule(period)` occurs at nonnegative integer multiples of `period`. `OffsetRegularSchedule(period, offset)` begins at `offset` and repeats at the given period. Times are stored exactly, so rational periods such as `1//10` are useful when exact event alignment matters.

```julia
function init(t, specs, seed)
    return ModelDescription(;
        discrete_states = (;
            count = 0,
        ),
        schedules = (;
            sample = RegularSchedule(1//10),
            delayed = OffsetRegularSchedule(1//2, 1//4),
        ),
    )
end

function updates(t, model)
    on_triggering(model.sample, t) do
        return UpdatesOutput(;
            updates = (;
                count = model.count + 1,
            ),
        )
    end
end
```

Initialization establishes the model at `t_start`; it does not run a discrete update there.

Note that `updates_fcn` will be called on _every_ sample. Each model can check whether its schedule [`is_triggering`](@ref) before performing the corresponding work. (Further, a model can have many schedules and determine what should be done on each sample based on all of its schedules.)

```@docs
SystemsOfSystems.RegularSchedule
SystemsOfSystems.OffsetRegularSchedule
```

#### Resources

Resources are external objects that must be opened before simulation and closed afterward, such as files, sockets, or library handles. These can be declared as variables in the `resources` field of a `ModelDescription`. The opened payload becomes a field of the model, and SystemsOfSystems closes it even if the simulation encounters an error.

`OutputFile` is the common case:

```julia
function init(t, specs, seed)
    return ModelDescription(;
        resources = (;
            events = OutputFile(;
                name = "events.csv",
            ),
        ),
    )
end
```

For a relative file name, [`SimOptions`](@ref) uses `outdir` as the top-level output directory. By default, an output file is scoped beneath directories matching its model path, which prevents identically named files from different submodels from colliding.

`Resource` wraps arbitrary open and close functions. The open function receives resource inputs followed by `open_args`; its return value is the payload stored on the model.

```julia
Resource(;
    open_args = (host, port),
    open_fcn = (inputs, host, port) -> open_connection(host, port),
    close_fcn = close,
)
```

Writing to a resource from `rates_fcn` can have unexpected results because rate calculations may be provisional. External side effects are better performed from discrete updates or another explicitly controlled part of the simulation.

```@docs
SystemsOfSystems.OutputFile
SystemsOfSystems.Resource
```

### Continuous-Time Dynamics

The continuous-time dynamics function is called as `rates_fcn(t, model)`. It returns a [`RatesOutput`](@ref) containing derivatives, continuous outputs, and the continuous-time results of any submodels. Every derivative must have the same type as its corresponding state, and the solver will integrate it over time to update the state, consistently with all other continuous-time variables in the simulation.

```julia
function rates(t, model::MyModel)
    return RatesOutput(;
        rates = (;
            position = model.velocity,
            velocity = -model.position / model.mass,
        ),
        outputs = (;
            energy = (model.position^2 + model.mass * model.velocity^2) / 2,
        ),
        models = (;
            sensor = sensor_rates(t, model.sensor),
        ),
    )
end
```

A `rates_fcn` can be evaluated several times during one solver step, including at intermediate Runge-Kutta stages and during rejected adaptive steps. Side effects are therefore unsafe: writing files, incrementing counters, or taking random draws inside it could occur an unexpected number of times. States, random variables, outputs, and resources provide the corresponding simulation-aware mechanisms.

A model with no continuous-time behavior can be omitted from its parent's `models` result. Likewise, a continuous state omitted from `rates` is held constant.

`RatesOutput` can set `stop = true` to end the simulation after the current accepted sample has been processed.

```@docs
SystemsOfSystems.RatesOutput
```

### Discrete-Time Dynamics

The discrete-time dynamics function is called as `updates_fcn(t, model)` after every accepted simulation step. It returns an [`UpdatesOutput`](@ref) containing state changes, discrete outputs, and the discrete-time results of any submodels.

```julia
function updates(t, model::MyModel)
    new_mode = choose_mode(model)
    return UpdatesOutput(;
        updates = (;
            mode = new_mode,
        ),
        outputs = (;
            mode_changed = new_mode != model.mode,
        ),
        models = (;
            sensor = sensor_updates(t, model.sensor),
        ),
    )
end
```

The result can be sparse. States and submodels that are omitted retain their prior values. If nothing changes at a sample, the function can return `nothing`. This is especially convenient with [`on_triggering`](@ref), which returns `nothing` when its schedule is not triggering.

`UpdatesOutput` can set `stop = true` to end the simulation after the current sample has been processed.

A continuous state may also be changed discontinuously by including it in the `updates` block. This is useful for resets, impacts, discontinuous mode changes, and similar hybrid dynamics.

```julia
function updates(t, model)
    if model.position <= 0 && model.velocity < 0
        return UpdatesOutput(;
            updates = (;
                position = 0.,
                velocity = -0.8 * model.velocity, # Bounce.
            ),
        )
    end
    return nothing
end
```

The update above creates a discontinuity. The next continuous-time evaluation starts from the new values.

```@docs
SystemsOfSystems.UpdatesOutput
```

### Unavailable Outputs

A model can return `missing` for a continuous or discrete output when no sample is
available at the current time. The logger skips that value and its timestamp.

When an output has no initial value, a `VariableDescription` can declare its eventual type
explicitly:

```julia
discrete_outputs = (;
    measurement = VariableDescription{Float64}(
        missing;
        title = "Measurement",
        dimensions = ["measurement" => "m"],
    ),
)
```

Skipping a value leaves no marker for the unavailable interval. Linear interpolation will
bridge the gap between surrounding samples, and sample-and-hold interpolation will return
the preceding value. The recorded timestamps indicate when the model actually supplied
values. If `missing` itself must be retained as output data, it can be wrapped in a distinct
value such as `Some(missing)` or represented by a model-specific type.

## Custom `t_next`

Schedules are best for declarative event patterns known at initialization. A model can use `t_next` when its next event time is dynamic, such as when the next event depends on a state, an input, or the outcome of the current update.

The first requested time can be set in [`ModelDescription`](@ref):

```julia
function init(t, specs, seed)
    return ModelDescription(;
        discrete_states = (;
            event_count = 0,
        ),
        t_next = t + specs.initial_delay,
    )
end
```

At the requested sample, the next time can be returned in `UpdatesOutput`:

```julia
function updates(t, model)
    return UpdatesOutput(;
        updates = (;
            event_count = model.event_count + 1,
        ),
        t_next = t + next_delay(model),
    )
end
```

The requested `t_next` is a hard upper bound for the integrator, just like a schedule occurrence or a user-provided time. It must be later than the current event if it is meant to create another future sample.

If `UpdatesOutput.t_next` is omitted, its default value, [`KEEP_T_NEXT`](@ref), retains the model's previous request. [`NO_T_NEXT`](@ref) can be used to cancel a pending request when the model has no next event.

```@docs
SystemsOfSystems.KEEP_T_NEXT
SystemsOfSystems.NO_T_NEXT
```
