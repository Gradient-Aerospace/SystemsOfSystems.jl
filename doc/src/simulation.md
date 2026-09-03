# Simulation

One function handles running the simulation: `simulate`.

```@docs
SystemsOfSystems.simulate
```

## Simulation Results

`simulate` returns one [`SimHistory`](@ref). It contains the requested start time, the last completed simulation time, the final model, the log, and the reason the simulation stopped.

```julia
history = simulate(...)

history.t_start
history.t_stop
history.model
history.stop
```

Julia's property destructuring is convenient when only part of the result is needed:

```julia
(; t_stop, model) = simulate(...)
```

[`succeeded`](@ref) reports whether the simulation ended normally. Reaching the requested end time and a deliberate stop request both count as success; an exception or numerical solver failure does not.

```julia
if !succeeded(history)
    @warn "Simulation failed" reason = history.stop
end
```

```@docs
SystemsOfSystems.SimHistory
SystemsOfSystems.succeeded
```

## Recorded Histories

`SimHistory` forwards the dictionary-like log interface, so users normally do not need to access its `log` field directly. These expressions are equivalent:

```julia
history["/vehicle/controller"]["command"]
history.log["/vehicle/controller"]["command"]
```

Each model path returns a [`Logs.ModelHistory`](@ref). Its constants, states, outputs, and submodels can be accessed by string or symbol. Logging policies may omit selected variables while preserving the model-history structure.

[`Logs.gather_all_time_series`](@ref) collects every recorded `TimeSeries` into one ordered dictionary. Its keys combine the model path and variable name, making it useful for searching, exporting, or passing a flat collection to another tool.

```julia
all_series = Logs.gather_all_time_series(history)
position = all_series["/vehicle:position"]
```

```@docs
SystemsOfSystems.Logs.ModelHistory
SystemsOfSystems.Logs.gather_all_time_series
```

## Time-Series Utilities

[`SystemsOfSystems.select`](@ref) derives a new time series while preserving its timestamps and metadata. It is public but qualified because `select` is a common name in data-analysis packages.

```julia
speed = SystemsOfSystems.select(history["/"]["velocity"]; title = "Speed") do velocity
    abs(velocity)
end
```

[`plot_ts`](@ref) creates a new Makie figure. [`plot_ts!`](@ref) adds a time series to an existing figure or layout target. Either function requires a loaded Makie backend.

```julia
using CairoMakie

figure = Figure()
plot_ts!(figure[1, 1], history["/"]["position"])
figure
```

```@docs
SystemsOfSystems.select
SystemsOfSystems.plot_ts
SystemsOfSystems.plot_ts!
```

## Stop Reasons

The `history.stop` field retains the specific reason the simulation ended. Normal stop reasons subtype `AbstractStopReason`, while exceptions and numerical failures subtype `AbstractFailureReason`. Most code can use `succeeded(history)` and only inspect the concrete reason when reporting or recovering from a failure.

```@docs
SystemsOfSystems.AbstractTerminationReason
SystemsOfSystems.AbstractStopReason
SystemsOfSystems.AbstractFailureReason
SystemsOfSystems.ReachedEndTime
SystemsOfSystems.ModelRequestedStop
SystemsOfSystems.HookRequestedStop
SystemsOfSystems.EncounteredError
SystemsOfSystems.Solvers.SolverFailedToConverge
SystemsOfSystems.Solvers.SolverStepSizeUnderflow
SystemsOfSystems.describe
```
