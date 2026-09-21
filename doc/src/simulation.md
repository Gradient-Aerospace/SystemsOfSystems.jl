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

[`succeeded`](@ref) reports whether the simulation ended normally. Reaching the requested end time, a deliberate stop request, and a catchable interruption count as success; an unexpected exception or numerical solver failure does not. Applications requiring completion can check for `ReachedEndTime` or their expected model stop reason.

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

The `history.stop` field retains the specific reason the simulation ended. Normal stop reasons subtype `AbstractStopReason`, while unexpected exceptions and numerical failures subtype `AbstractFailureReason`.

### Interruption

A catchable `InterruptException` raised during the simulation loop produces `Interrupted(history.t_stop)`. The returned time and model describe the last fully accepted simulation sample, including its discrete update. An interrupted intermediate solver stage is discarded. Hooks and resources follow the normal teardown path and receive the accepted endpoint.

Interruption leaves logging unchanged, just as an unexpected exception does. No final sample is added and no new continuous outputs are evaluated. With regular sampling, the last logged state may therefore be older than `history.t_stop`; `history.model` contains the last accepted state.

`succeeded(history)` is true for `Interrupted`: this reports a normal engine lifecycle, not functional completion. For example, a workflow requiring the requested end time can test `history.stop isa SystemsOfSystems.ReachedEndTime`.

SystemsOfSystems handles catchable Julia interruptions during the simulation loop. Applications control operating-system signal handling and process exit codes. Uncatchable termination cannot guarantee cleanup or complete log files.

### Reference

```@docs
SystemsOfSystems.AbstractTerminationReason
SystemsOfSystems.AbstractStopReason
SystemsOfSystems.AbstractFailureReason
SystemsOfSystems.ReachedEndTime
SystemsOfSystems.ModelRequestedStop
SystemsOfSystems.HookRequestedStop
SystemsOfSystems.Interrupted
SystemsOfSystems.EncounteredError
SystemsOfSystems.Solvers.SolverFailedToConverge
SystemsOfSystems.Solvers.SolverStepSizeUnderflow
SystemsOfSystems.describe
```
