# Simulation

One function handles running the simulation: `simulate`.

```@docs
SystemsOfSystems.simulate
```

## Simulation Results

`simulate` returns one [`SimHistory`](@ref). It contains the requested start time, the last completed simulation time, the final model, the log, the reason the simulation stopped, and any failures while closing hooks or resources.

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

[`succeeded`](@ref) reports whether the simulation ended without a failure: it is equivalent to `!failed(history.stop)`. Reaching the requested end time, a deliberate stop request, and a catchable interruption count as success; an unexpected exception or numerical solver failure does not. [`finished`](@ref SystemsOfSystems.finished) distinguishes a nominal end condition from an interruption, but includes deliberate early stops. These functions work for both a newly returned history and a loaded history.

Cleanup failures are independent: `history.cleanup_errors` lists them in close order, with the hook or resource in `context` and the original exception and trace in `exception` and `trace`. Applications requiring finalized outputs can check `isempty(history.cleanup_errors)` in addition to `succeeded(history)`. Loaded cleanup errors retain their context and diagnostic text, rather than their exception objects and traces.

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

## Stop Reasons

The history returned from `simulate` includes a `stop` field which contains an `AbstractTerminationReason`, which has the following interface:

| Function | Result |
| --- | --- |
| `finished(reason)::Bool` | Whether the simulation reached a nominal end condition, including a model or hook stop request. |
| `failed(reason)::Bool` | Whether an unexpected exception or numerical failure prevented propagation. |
| `interrupted(reason)::Bool` | Whether the simulation was interrupted. |
| `describe(reason)::String` | A concise description of why the simulation stopped. |
| `details(reason)::String` | Additional diagnostic text, or an empty string. |

For example, we can report a run's status without depending on its concrete reason type:

```julia
using SystemsOfSystems.TerminationReasons

println(describe(history.stop))
if interrupted(history.stop)
    println("The run was interrupted before a nominal end condition.")
elseif failed(history.stop)
    println(details(history.stop))
end
```

Custom reasons can subtype `AbstractTerminationReason` and provide custom methods for the interface functions above.

Applications requiring a particular end condition can inspect the concrete reason in a history returned by `simulate`, for example with `history.stop isa ReachedEndTime`. `finished` reports nominal termination generally, so it does not distinguish reaching the requested end time from stopping early at a model's request. The summary's type name and descriptions are descriptive text rather than a stable machine-readable classification of specific reasons.

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
│   ├── sim_history_version = 1
│   ├── systems_of_systems_version = "..."
│   ├── log_path = "/log"
│   ├── t_start = 0.0
│   ├── t_stop = 1.0
│   ├── stop/
│   │   ├── type = "SystemsOfSystems.TerminationReasons.ReachedEndTime"
│   │   ├── finished = true
│   │   ├── failed = false
│   │   ├── interrupted = false
│   │   ├── summary = "The sim reached the specified end time of 1.0."
│   │   └── details = ""
│   ├── cleanup_errors/
│   │   └── count = 0             # Number of failed hook or resource closes
│   └── model/ ...                # Optional final model value
└── log/
    ├── log_format_version = 1
    ├── systems_of_systems_version = "..."
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

When logging was disabled, `/log` contains `is_null = true` and its format/provenance datasets, with no model tree. History and log format versions are checked independently when loading; the [format compatibility policy](hdf5_format.md#Format-Versions-and-Compatibility) describes supported versions and legacy files.

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

For a file containing several runs or other application data, the group-based methods leave file ownership with the caller. Relative paths are resolved within the supplied parent, and the default log path is `"log"` within that parent. For example, `save_sim_history(run_group, "history", history)` puts both groups inside `run_group`. Absolute paths still select locations from the file root:

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

### Unsuccessful Saves

Saving replaces the destination contents and is not transactional. If a write fails, the destination may contain a partial result, and previous contents may already have been replaced. This includes failures while saving an optional model. When an existing result needs to be preserved until a new save succeeds, the new result can be saved to a separate file first. Saving metadata beside a live log preserves that log, but failed metadata writes may leave an incomplete history group.

### Termination Summaries

In general, termination reasons (`<: AbstractTerminationReason`) can be flexible and store information that cannot always be meaningfully saved and loaded. When saving a `SimHistory`, this information is discarded, and a [`TerminationSummary`](@ref SystemsOfSystems.TerminationSummary) will be saved in its stead. The `TerminationSummary` stores the results of the API for termination reasons (`finished`, `failed`, `interrupted`, `describe`, and `details`), and hence it can be loaded and return the same values for those functions. However, the specific type information for a termination reason is lost on saving.

The original type name is retained in the summary's `type` field for identification. Reason-specific fields, such as a model path or solver step size, are available only through the saved descriptions when those descriptions include them. Hooks, exceptions, and stack traces are not restored. For `EncounteredError`, `details` retains the rendered exception and stack trace as text.

The same summary is useful without saving a history. For example, we can collect portable information for another application:

```julia
using SystemsOfSystems.TerminationReasons

summary = TerminationSummary(history.stop)
summary.type
summary.summary
summary.details
finished(summary) == finished(history.stop) # true
```

```@docs
SystemsOfSystems.save_sim_history
SystemsOfSystems.load_sim_history
SystemsOfSystems.TerminationSummary
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

### Interruption

A catchable `InterruptException` raised during the simulation loop produces `Interrupted(history.t_stop)`. The returned time and model describe the last fully accepted simulation sample, including its discrete update. An interrupted intermediate solver stage is discarded. Hooks and resources follow the normal teardown path and receive the accepted endpoint.

Interruption leaves logging unchanged, just as an unexpected exception does. No final sample is added and no new continuous outputs are evaluated. With regular sampling, the last logged state may therefore be older than `history.t_stop`; `history.model` contains the last accepted state.

For an interruption, `interrupted(history.stop)` is true, while `finished(history.stop)` and `failed(history.stop)` are false. `succeeded(history)` remains true because it reports the absence of failure rather than functional completion. These results are preserved when saving and loading the history.

SystemsOfSystems handles catchable Julia interruptions during the simulation loop. Applications control operating-system signal handling and process exit codes. Uncatchable termination cannot guarantee cleanup or complete log files.

### Reference

```@docs
SystemsOfSystems.AbstractTerminationReason
SystemsOfSystems.AbstractStopReason
SystemsOfSystems.AbstractFailureReason
SystemsOfSystems.AbstractInterruption
SystemsOfSystems.ReachedEndTime
SystemsOfSystems.ModelRequestedStop
SystemsOfSystems.HookRequestedStop
SystemsOfSystems.Interrupted
SystemsOfSystems.EncounteredError
SystemsOfSystems.Solvers.SolverFailedToConverge
SystemsOfSystems.Solvers.SolverStepSizeUnderflow
SystemsOfSystems.describe
SystemsOfSystems.finished
SystemsOfSystems.failed
SystemsOfSystems.interrupted
SystemsOfSystems.details
```
