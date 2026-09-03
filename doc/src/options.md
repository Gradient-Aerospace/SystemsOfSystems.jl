# Options

The [`simulate`](@ref) function accepts a [`SimOptions`](@ref) value that controls output paths, logging, the continuous-time solver, hooks, and the time label used in plots.

```julia
history = simulate(
    user_data;
    t = (0, 10),
    init_fcn,
    rates_fcn,
    updates_fcn,
    options = SimOptions(;
        outdir = "out",
        solver = Solvers.DormandPrince54Options(),
        hooks = [
            Hooks.ProgressBarOptions(),
        ],
        log = Logs.BasicLogOptions(),
        time_dimension = "Time" => "s",
    ),
)
```

Every field has a default. Most simulations can begin with `SimOptions()` or omit the `options` keyword entirely.

```@docs
SystemsOfSystems.SimOptions
```

## Solvers

The solver advances continuous states between event times. The default is the adaptive Dormand-Prince 5(4) solver.

`DormandPrince54Options` is appropriate for most simulations. `initial_dt` is its first proposed step, `max_dt` limits later proposals, and `abs_tol` and `rel_tol` control local error. Accepted steps will be shortened when necessary to stop exactly at a user-requested time, schedule occurrence, or model `t_next`.

```julia
options = SimOptions(;
    solver = Solvers.DormandPrince54Options(;
        initial_dt = 0.01,
        max_dt = 0.1,
        abs_tol = 1e-6,
        rel_tol = 1e-5,
    ),
)
```

When it's unnecessary for the solver to adapt the step, then an explicit Runge-Kutta solver like `RungeKutta4Options` is faster. Its `dt` is essentially the maximum time step it will take, but it can be cut short in order to step exactly to scheduled times, model-requested `t_next` values, etc.

```julia
options = SimOptions(;
    solver = Solvers.RungeKutta4Options(;
        dt = 1//100,
    ),
)
```

The times passed as `t` to `simulate` are required sample times, not a general fixed-step setting. For example, `t = 0:1:10` makes the solver stop at every whole second, but an adaptive solver can take as many smaller steps as necessary between those times.

```@docs
SystemsOfSystems.Solvers.DormandPrince54Options
SystemsOfSystems.Solvers.RungeKutta4Options
SystemsOfSystems.Solvers.Ralston2Options
```

### Structured-State Errors

The adaptive solver must reduce the error in each state variable to one normalized scalar. The default [`SystemsOfSystems.normalized_variable_error`](@ref) implementation compares the components returned by `Dimensions.eachdim` and returns the largest error. This works automatically for scalars, arrays, static arrays, and user types that implement the Dimensions interface.

A state type that does not use Dimensions can specialize the function directly. For example:

```julia
struct PositionVelocity
    position::Float64
    velocity::Float64
end

function SystemsOfSystems.normalized_variable_error(
    value::PositionVelocity,
    embedded_value::PositionVelocity,
    absolute_tolerance,
    relative_tolerance,
)
    return max(
        SystemsOfSystems.normalized_scalar_error(
            value.position,
            embedded_value.position,
            absolute_tolerance,
            relative_tolerance,
        ),
        SystemsOfSystems.normalized_scalar_error(
            value.velocity,
            embedded_value.velocity,
            absolute_tolerance,
            relative_tolerance,
        ),
    )
end
```

These functions are public but intentionally qualified because most users never call them directly.

```@docs
SystemsOfSystems.normalized_variable_error
SystemsOfSystems.normalized_scalar_error
```

### Custom Solvers

Solver extensions define an immutable `Solvers.AbstractSolverOptions`, a per-simulation `Solvers.AbstractIntegrator`, and methods for `Solvers.create_integrator` and `Solvers.step!`. A wrapper can add behavior to an existing solver without reimplementing its numerical method:

```julia
struct CountingSolverOptions{O} <: Solvers.AbstractSolverOptions
    solver::O
    count::Base.RefValue{Int}
end

struct CountingIntegrator{I} <: Solvers.AbstractIntegrator
    integrator::I
    count::Base.RefValue{Int}
end

function Solvers.create_integrator(options::CountingSolverOptions, problem, initial_state)
    integrator = Solvers.create_integrator(options.solver, problem, initial_state)
    return CountingIntegrator(integrator, options.count)
end

function Solvers.step!(integrator::CountingIntegrator, problem, request)
    integrator.count[] += 1
    return Solvers.step!(integrator.integrator, problem, request)
end
```

A solver implemented from scratch receives one `Solvers.StepRequest` at a time and returns either `Solvers.AcceptedStep` or `Solvers.SolverFailure`. These protocol types are public but remain qualified under `Solvers`.

```@docs
SystemsOfSystems.Solvers.AbstractSolverOptions
SystemsOfSystems.Solvers.AbstractIntegrator
SystemsOfSystems.Solvers.StepRequest
SystemsOfSystems.Solvers.AcceptedStep
SystemsOfSystems.Solvers.SolverFailure
SystemsOfSystems.Solvers.create_integrator
SystemsOfSystems.Solvers.step!
```

## Hooks

Hooks allow other processes to interact with the simulation loop, and the sim can have any number of hooks in the `hooks` vector of `SimOptions`.

```julia
options = SimOptions(;
    hooks = [
        Hooks.ProgressBarOptions(),
        Hooks.SimTimeoutOptions(;
            max_run_time = 60.,
        ),
    ],
)
```

`ProgressBarOptions` displays command-line progress. Its update interval is wall-clock time, not simulation time.

`SimTimeoutOptions` requests a clean stop after the simulation has run for the specified wall-clock duration. The timeout is checked by the simulation loop; it is not a hard operating-system deadline that interrupts arbitrary user code.

`ClockSyncOptions` prevents the simulation from advancing faster than wall-clock time. It is useful for soft real-time demonstrations and hardware- or software-in-the-loop setups. It cannot make a simulation run in real time when one simulation step requires more computation than the corresponding wall-clock interval.

These are the built-in hooks.

```@docs
SystemsOfSystems.Hooks.ProgressBarOptions
SystemsOfSystems.Hooks.SimTimeoutOptions
SystemsOfSystems.Hooks.ClockSyncOptions
```

Further hooks can be developed using the hooks interface.

A custom hook normally defines separate option and runtime types. The creation method receives the requested simulation times and initial model; update methods receive each accepted time and the corresponding pre-update model. The default `Hooks.close_hook!` does nothing, so a hook only needs to specialize it when cleanup is necessary.

```julia
struct CallbackHookOptions{F} <: Hooks.AbstractHookOptions
    callback::F
end

struct CallbackHook{F} <: Hooks.AbstractHook
    callback::F
end

Hooks.create_hook(options::CallbackHookOptions, t, model) =
    CallbackHook(options.callback)

function Hooks.update_hook!(hook::CallbackHook, t, model)
    hook.callback(t, model)
    return Hooks.HookOutputs()
end
```

```@docs
SystemsOfSystems.Hooks.AbstractHookOptions
SystemsOfSystems.Hooks.AbstractHook
SystemsOfSystems.Hooks.HookOutputs
SystemsOfSystems.Hooks.create_hook
SystemsOfSystems.Hooks.update_hook!
SystemsOfSystems.Hooks.close_hook!
```

## Logs

The `log` option selects where, and whether, simulation histories are stored.

`BasicLogOptions` is the default. It stores selected histories in ordinary Julia arrays in memory and is normally the fastest choice.

```julia
options = SimOptions(;
    log = Logs.BasicLogOptions(),
)
```

`NullLogOptions` turns time-series logging off. It is useful when only fields such as `history.t_stop` and `history.model` are needed.

```julia
options = SimOptions(;
    log = Logs.NullLogOptions(),
)
```

`HDF5LogOptions` writes time-series data directly to disk. This supports histories that are too large for RAM, at the cost of slower simulation. Constants that cannot be represented by HDF5Vectors are omitted with a warning that identifies the constant, its type, and the underlying error. HDF5 logging becomes available after importing `HDF5Vectors`.

```julia
using HDF5Vectors

options = SimOptions(;
    log = Logs.HDF5LogOptions(;
        filename = "out/history.h5",
    ),
)
```

If the history fits in memory and only the final artifact needs to be HDF5, a `BasicLogOptions` simulation followed by `Logs.save_log_to_hdf5` is faster than logging directly to HDF5. The same unsupported-constant behavior applies when saving an existing log.

The HDF5 representation retains model order and types; constants and their `VariableDescription` metadata; and each time series' title, dimensions, signal path, continuous/discrete designation, groups, and interpolator. Model types and interpolators use Julia serialization. Files should therefore come only from trusted sources, and custom serialized types must be available when loading.

`Logs.load_hdf5_log` returns `(log, root_model_history)`. The returned time series remain backed by the open file, so the log should be closed when it is no longer needed:

```julia
log, root = Logs.load_hdf5_log("out/history.h5")
try
    position = root["position"]
    # Use the loaded history.
finally
    Logs.close_log(log)
end
```

```@docs
SystemsOfSystems.Logs.BasicLogOptions
SystemsOfSystems.Logs.NullLogOptions
SystemsOfSystems.Logs.HDF5LogOptions
SystemsOfSystems.Logs.load_hdf5_log
SystemsOfSystems.Logs.save_log_to_hdf5
```

### Standalone Time Series

Individual time series can use the same HDF5 representation without constructing a complete log. These functions operate on an open HDF5 file, and loaded vectors remain usable only while that file is open.

```julia
using HDF5
using HDF5Vectors

HDF5.h5open("signal.h5", "w") do file
    Logs.save_time_series_to_hdf5(file, "signals/position", position)
end

HDF5.h5open("signal.h5", "r") do file
    loaded_position = Logs.load_time_series_from_hdf5(file, "signals/position")
    # Use loaded_position before this block closes the file.
end
```

```@docs
SystemsOfSystems.Logs.save_time_series_to_hdf5
SystemsOfSystems.Logs.load_time_series_from_hdf5
SystemsOfSystems.Logs.close_log
```

### Logging Policies

Both `BasicLogOptions` and `HDF5LogOptions` accept a `logging_policy`. A logging policy assigns two choices to each model:

* A variable set that selects _which_ constants, states, and outputs are stored
* A sampler that selects _when_ the states and outputs are recorded

The default `AllPassLoggingPolicy` stores every variable from every model and samples at every simulation time.

```@docs
SystemsOfSystems.LoggingPolicies.AllPassLoggingPolicy
```

#### One Policy for Every Model

`UniformLoggingPolicy` applies the same `ModelLoggingPolicy` to all models. For example, the following stores all variables but records states and outputs only on times that align with a 0.1-second grid:

```julia
using SystemsOfSystems: LoggingPolicies, Samplers

logging_policy = LoggingPolicies.UniformLoggingPolicy(;
    policy = LoggingPolicies.ModelLoggingPolicy(;
        sampler = Samplers.RegularSampler(1//10),
    ),
)

options = SimOptions(;
    log = Logs.BasicLogOptions(; logging_policy),
)
```

A logging sampler does not force the simulation to take steps. (Changing the log never affects the result of the simulation.) It only selects from times that already exist.

```@docs
SystemsOfSystems.LoggingPolicies.UniformLoggingPolicy
SystemsOfSystems.LoggingPolicies.ModelLoggingPolicy
SystemsOfSystems.Samplers.CompleteSampler
SystemsOfSystems.Samplers.NullSampler
SystemsOfSystems.Samplers.RegularSampler
```

#### Policies by Model Path

`RegexLoggingPolicy` allows the user to provide different model logging policies to different models according to the models' "paths". The root model has path `"/"`, while descendants have paths such as `"/plant"` and `"/vehicle/sensor"`. The first matching rule wins.

The following policy selects plant samples on a 100 Hz grid, omits two variables from the controller, and stores every variable from all remaining models:

```julia
logging_policy = LoggingPolicies.RegexLoggingPolicy(;
    rules = [
        r"^/plant$" => LoggingPolicies.ModelLoggingPolicy(;
            sampler = Samplers.RegularSampler(1//100),
        ),
        r"^/controller$" => LoggingPolicies.ModelLoggingPolicy(;
            variable_set = LoggingPolicies.VariableExclusionList([
                :large_cache,
                :debug_state,
            ]),
        ),
    ],
    default = LoggingPolicies.AllPassModelLoggingPolicy(),
)

options = SimOptions(;
    log = Logs.BasicLogOptions(; logging_policy),
)
```

Again, for the sake of clarity, note that asking for sampling at every 1/100 period does _not_ force the simulation to take steps that align with that sampling period. (Again, we do not want the log type to influence the results of the simulation.) The selected sampler therefore needs to align with the discrete steps requested by schedules, `t_next`, etc.

If no `default` is specified for `RegexLoggingPolicy`, models that don't match the regular expressions are omitted from the log (they receive a `NullModelLoggingPolicy`).

```@docs
SystemsOfSystems.LoggingPolicies.RegexLoggingPolicy
SystemsOfSystems.LoggingPolicies.AllPassModelLoggingPolicy
SystemsOfSystems.LoggingPolicies.NullModelLoggingPolicy
```

#### Selecting Variables

`ModelLoggingPolicy.variable_set` controls which variables are present in that model's history:

* `AllVariables()` selects every variable.
* `NoVariables()` selects no variables.
* `VariableList(names)` selects only the listed variables.
* `VariableExclusionList(names)` selects everything except the listed variables.

This can be useful for models that have "weird" states. E.g., a discrete state could be a function, but we might not want to log a time history of functions (though we could).

Names can be strings or symbols. A model's variable set does not control its submodels; each submodel receives its own model logging policy.

```@docs
SystemsOfSystems.LoggingPolicies.AllVariables
SystemsOfSystems.LoggingPolicies.NoVariables
SystemsOfSystems.LoggingPolicies.VariableList
SystemsOfSystems.LoggingPolicies.VariableExclusionList
```
