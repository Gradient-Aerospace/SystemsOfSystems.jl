# Simulation Time

SystemsOfSystems distinguishes exact, official simulation times from the floating-point times used for numerical integration. User-requested times, schedule occurrences, and model-requested `t_next` values are official times. This allows unrelated periodic systems to remain aligned without accumulating floating-point roundoff errors from step to step.

The numerical solver evaluates `rates_fcn` at floating-point times between official samples. In contrast, `init_fcn`, `updates_fcn`, schedules, and `t_next` use exact times at the simulation boundary.

## ExactTime

`ExactTime` is the rational representation used for official simulation times. Values supplied as integers, rationals, or floating-point numbers are converted to this representation when they enter the scheduler. Rational values such as `1//10` are useful when an event time must be represented explicitly and exactly.

The `exact_time` function performs this conversion. Floating-point inputs use Julia's `rationalize` behavior, which recovers simple values such as converting `0.1` to `1//10`. The type and conversion function live in the `SimulationTimes` module because most models can rely on automatic conversion.

```@docs
SystemsOfSystems.SimulationTimes.ExactTime
SystemsOfSystems.SimulationTimes.exact_time
```

Two special exact-time values are used with a dynamic `ModelDescription.t_next`:

* [`SystemsOfSystems.KEEP_T_NEXT`](@ref) retains the model's previous request when returned from `UpdatesOutput`.
* [`SystemsOfSystems.NO_T_NEXT`](@ref) indicates that the model has no finite upcoming event.

These sentinels are public but not exported because their meaning depends on the simulation-time interface. Model code can refer to them with the `SystemsOfSystems.` prefix. `KEEP_T_NEXT` is the default when `UpdatesOutput.t_next` is omitted, while `NO_T_NEXT` is useful for cancelling a finite request explicitly:

```julia
UpdatesOutput(; t_next = SystemsOfSystems.NO_T_NEXT)
```

## Triggering

The `updates_fcn` is called on every step, and the step may result from a schedule, a model's `t_next`, the integrator's adaptive step method, etc. A schedule's [`is_triggering`](@ref) function indicates whether the current official time is one of that schedule's occurrences.

```julia
function updates(t, model)
    if is_triggering(model.sample_schedule, t)
        return UpdatesOutput(;
            updates = (;
                count = model.count + 1,
            ),
        )
    end
    return nothing
end
```

[`on_triggering`](@ref) provides the same check in Julia's `do`-block form. It evaluates the block when the schedule is triggering and otherwise returns `nothing`, which is also a valid result from `updates_fcn`.

```julia
function updates(t, model)
    on_triggering(model.sample_schedule, t) do
        return UpdatesOutput(;
            updates = (;
                count = model.count + 1,
            ),
        )
    end
end
```

Initialization establishes the model at `t_start`; it does not call `updates_fcn` there. Schedule queries apply to the accepted samples that follow initialization.

```@docs
SystemsOfSystems.is_triggering
SystemsOfSystems.on_triggering
```

## Finding Future Times

[`next_trigger_time`](@ref) returns the first occurrence of a schedule strictly later than a given official time. SystemsOfSystems uses this function internally when combining the next occurrences from every declared schedule.

[`next_regular_time`](@ref) provides the corresponding calculation for a regular sequence described directly by a period and offset. It can be useful for a dynamic `t_next` implementation whose next event follows a regular clock but is not declared in the model's `schedules`.

```julia
function updates(t, model)
    return UpdatesOutput(;
        updates = (;
            count = model.count + 1,
        ),
        t_next = next_regular_time(t, model.period, model.offset),
    )
end
```

The lower-level [`Schedules.is_regular_step_triggering`](@ref) function tests membership in the same periodic sequence without constructing a schedule. A `RegularSchedule` or `OffsetRegularSchedule` is generally more convenient for model code because it also requests the necessary simulation times. This function exists for backwards compatibility and is public but not exported.

```@docs
SystemsOfSystems.next_trigger_time
SystemsOfSystems.next_regular_time
SystemsOfSystems.Schedules.is_regular_step_triggering
```
