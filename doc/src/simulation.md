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

## Saving and Loading Histories

[`save_sim_history`](@ref) saves a record of a run to a single HDF5 file. [`load_sim_history`](@ref) restores its times, termination information, and a disk-backed log. Loading does not read all of the log into memory, so we can select just the series or samples needed for an analysis.

```julia
using SystemsOfSystems, HDF5Vectors

save_sim_history("history.h5", history)
loaded = load_sim_history("history.h5")
position = loaded["/vehicle"]["position"].data[:]
Logs.close_log(loaded.log)
```

For an analysis that needs the file only within a block, the filename-based `do` form closes it automatically, including when the block throws an exception:

```julia
position = load_sim_history("history.h5") do loaded
    collect(loaded["/vehicle"]["position"].data)
end
```

The call returns the block's result. Copied samples and computed statistics remain usable afterward; a returned history or time series that still depends on the file does not. Loading options such as `history_path` and `load_model` are also accepted. For caller-owned groups, an enclosing `HDF5.h5open(...) do fid` block can manage the file's lifetime instead.

The caller owns the loaded log's file and can release it with `Logs.close_log`. A history produced with logging disabled restores a `NullLog` and keeps no file open. Open HDF5 handles belong to their process; a loaded history should not be sent from a worker to another process. It can instead be loaded in the process that will use it.

### File Layout and Models

The [HDF5 file format](hdf5_format.md) documents the public entries for external readers, including model and time-series paths and sample encodings. Readers can rely on those entries while ignoring additional implementation-specific data. The abbreviated tree below also shows some Julia reconstruction entries, identified as such; it is not an exhaustive list of public keys.

The default groups are `/history` for run information and `/log` for logged data. The history's `log_path` dataset identifies the log group within the same HDF5 file. Start and stop times are ordinary floating-point datasets; loading restores them with `exact_time`, so it does not preserve distinctions lost in conversion to floating point. `Logs.load_hdf5_log` can read the log independently of the history metadata.

For example, a run from 0 to 1 second with a root state named `x` and a child model named `child` has the following layout. HDF5Vectors storage details are abbreviated as `...`.

```text
history.h5
├── history/
│   ├── @sim_history_version = 1
│   ├── @systems_of_systems_version = "..."
│   ├── log_path = "/log"
│   ├── t_start = 0.0
│   ├── t_stop = 1.0
│   ├── stop/
│   │   ├── type = "SystemsOfSystems.ReachedEndTime"
│   │   ├── description = "The sim reached the specified end time of 1.0."
│   │   ├── is_failure = false
│   │   ├── details = ...          # Present for records with diagnostic text
│   │   └── value/ ...            # Structured termination record
│   └── model/ ...                # Optional final model value
└── log/
    ├── @log_format_version = 1
    ├── @systems_of_systems_version = "..."
    ├── type = "Nothing"
    ├── serialized_type = ...     # Julia type reconstruction
    ├── constants/
    ├── names/ ...                # Names and ordering of logged variables
    ├── timeseries/
    │   └── x/
    │       ├── title = ...
    │       ├── path = "/x"
    │       ├── time/ ...
    │       ├── data/ ...
    │       └── ...               # Units, dimensions, interpolation, etc.
    └── models/child/ ...         # Each child repeats this log layout
```

When logging was disabled, `/log` contains `is_null = true` and its format/provenance attributes, with no model tree. History and log format versions are checked independently when loading; the [format compatibility policy](hdf5_format.md#Format-Versions-and-Compatibility) describes supported versions and legacy files.

The final model is omitted by default. If a model is suitable for storage with HDF5Vectors, we can opt into saving and loading it separately:

```julia
save_sim_history("history.h5", history; save_model = true)
loaded = load_sim_history("history.h5"; load_model = true)
```

The model is `nothing` when it was not saved or loading was not requested. Saving an unsupported model reports an error. Stored model types must be available when loading, and a model with process-local resources is not suitable for restoration merely because its fields can be serialized. As with saved logs, files should come from trusted sources because model and log metadata may use Julia serialization.

### Writing Logs and History to the Same File

An HDF5 simulation log starts in `/log`, so the file can accumulate samples during a run and receive its history metadata afterward:

```julia
filename = "flight.h5"
options = SimOptions(; log = Logs.HDF5LogOptions(filename))
history = simulate(model; init_fcn, rates_fcn, t = (0, 10), options)
save_sim_history(filename, history)
Logs.close_log(history.log)
```

Saving adds `/history` without moving or copying `/log`. Saving again replaces the history metadata; for example, saving with `save_model = false` removes a previously saved model. Existing log dataset handles remain usable. Saving to a different filename overwrites that file and copies the log through HDF5 without loading all its samples into memory. A history loaded read-only can be saved to another file, but cannot overwrite its own open file.

### Custom Groups and Results Containers

The default group names can be changed when saving and loading:

```julia
save_sim_history("results.h5", history;
    history_path = "/runs/first/history", log_path = "/runs/first/log")
loaded = load_sim_history("results.h5"; history_path = "/runs/first/history")
Logs.close_log(loaded.log)
```

Only the history path is needed when loading: its `log_path` dataset locates the log. This is a path within the HDF5 file, so renaming or moving the file does not break it. Direct-to-disk simulations can select their log location with `Logs.HDF5LogOptions(; filename, path = "/samples")`; saving the history to that file uses the existing log location by default.

For a file containing several runs or other application data, the group-based methods leave file ownership with the caller:

```julia
import HDF5

HDF5.h5open("results.h5", "w") do fid
    save_sim_history(fid, "/runs/first/history", history;
        log_path = "/runs/first/log")

    # Explicit groups offer the same interface without constructing paths in the saver.
    hg = HDF5.create_group(fid, "/runs/second/history")
    lg = HDF5.create_group(fid, "/runs/second/log")
    save_sim_history(hg, history; log_group = lg)
    loaded = load_sim_history(hg)
    samples = loaded["/"]["x"].data[:]
    Logs.close_log(loaded.log) # The caller's file and groups remain open.
    close(hg)
    close(lg)
end
```

The history and log groups must be in the same file and must not contain one another. Saving replaces their contents, except that a log already at its destination is preserved. Other groups in the file are untouched. A log loaded through a group remains usable only while the caller's file is open. `Logs.save_log_to_hdf5` and `Logs.load_hdf5_log` likewise accept a group, or a parent file/group and path, for working with logs independently.

Older standalone logs stored their data at the file root. `Logs.load_hdf5_log` still reads those files. A history using such a log can be saved to a new file with the current layout; saving metadata into that same root log would overlap its storage and is rejected.

### Termination Records

`history/stop/type`, `history/stop/description`, and `history/stop/is_failure` are ordinary HDF5 datasets that can be inspected without Julia. `history/stop/value` stores the restorable reason through HDF5Vectors. Simple built-in reasons (`ReachedEndTime`, `ModelRequestedStop`, `Interrupted`, the unknown-stop sentinel, and numerical solver failures) retain their concrete types and fields.

Hooks, exceptions, and custom termination reasons may hold live objects that do not belong in a saved record. These load as [`SystemsOfSystems.RecordedStop`](@ref) or [`SystemsOfSystems.RecordedFailure`](@ref), retaining the original type name and description. The failure distinction preserves `succeeded(history)`. Custom reason authors do not need to implement a persistence interface; a useful `describe` method provides the recorded description.

For `EncounteredError`, the recorded failure's `details` field contains the rendered exception and stack trace. The same text is available at `history/stop/details` for direct HDF5 inspection. Neither the exception object nor its compiler state is reconstructed. Re-saving a recorded reason retains its original identity and diagnostic text.

```@docs
SystemsOfSystems.save_sim_history
SystemsOfSystems.load_sim_history
SystemsOfSystems.RecordedStop
SystemsOfSystems.RecordedFailure
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
