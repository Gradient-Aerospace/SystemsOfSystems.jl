module TestSimHistoryPersistence

using Test
using SystemsOfSystems
using SystemsOfSystems: Logs, Solvers, Hooks, exact_time, describe,
    ReachedEndTime, ModelRequestedStop, HookRequestedStop, Interrupted, EncounteredError,
    RecordedStop, RecordedFailure, AbstractStopReason, AbstractFailureReason
import HDF5
import HDF5Vectors

# Use both a root model and a child to exercise relative storage paths. Fractional start
# and stop times also expose the intended conversion through Float64 in the saved record.
function small_history(; log = Logs.BasicLogOptions())
    return simulate(
        nothing;
        t = (1//3, 2//3),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 1.0),
            models = (; child = ModelDescription(; continuous_states = (; y = 2.0))),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.0),
            models = (; child = RatesOutput(; rates = (; y = -1.0))),
        ),
        options = SimOptions(; log),
    )
end

function with_stop(history, stop)
    return SimHistory(history.t_start, history.t_stop, history.log, history.model, stop)
end

# These reasons deliberately retain a running task, which cannot be reconstructed as part
# of a saved run. Their round trips should use descriptions without serializing the task.
struct ResourceHook <: Hooks.AbstractHook
    task::Task
end
Base.show(io::IO, ::ResourceHook) = print(io, "ResourceHook")

struct CustomStop <: AbstractStopReason
    task::Task
end
struct CustomFailure <: AbstractFailureReason
    task::Task
end
SystemsOfSystems.describe(::CustomStop) = "A custom stop."
SystemsOfSystems.describe(::CustomFailure) = "A custom failure."

@testset "History file layout, lazy logs, and optional models" begin

    # Saving an in-memory log separates metadata from samples. The model is opt-in,
    # and times and termination labels should be readable using only HDF5.
    history = small_history()
    filename = joinpath(mktempdir(), "history.h5")
    @test isnothing(save_sim_history(filename, history))
    HDF5.h5open(filename, "r") do fid
        @test Set(keys(fid)) == Set(["history", "log"])
        @test read(fid["history/log_path"]) == "/log"
        @test read(HDF5.attributes(fid["history"])["sim_history_version"]) == 1
        @test read(fid["history/t_start"]) === Float64(history.t_start)
        @test read(fid["history/t_stop"]) === Float64(history.t_stop)
        @test read(fid["history/stop/type"]) == string(typeof(history.stop))
        @test read(fid["history/stop/description"]) == describe(history.stop)
        @test !read(fid["history/stop/is_failure"])
        @test haskey(fid, "log/models/child")
        @test !haskey(fid, "history/model")
    end

    # Loading restores the recorded times and model hierarchy, but leaves sample vectors
    # on disk. Asking for an unsaved model should simply return nothing.
    restored = load_sim_history(filename; load_model = true)
    @test restored.t_start == exact_time(Float64(history.t_start))
    @test restored.t_stop == exact_time(Float64(history.t_stop))
    @test restored.stop == history.stop
    @test isnothing(restored.model)
    @test restored["/"]["x"].data isa HDF5Vectors.HDF5Vector
    @test restored["/"]["x"].data[:] == history["/"]["x"].data
    @test restored["/child"]["y"].data[:] == history["/child"]["y"].data
    Logs.close_log(restored.log)

    # Saving the model does not make loading it mandatory. Check both choices against the
    # same file so opting out cannot accidentally depend on the model entry being absent.
    save_sim_history(filename, history; save_model = true)
    restored = load_sim_history(filename)
    @test isnothing(restored.model)
    Logs.close_log(restored.log)
    restored = load_sim_history(filename; load_model = true)
    @test restored.model == history.model
    Logs.close_log(restored.log)

    # Replace the model with an invalid encoding to prove the default loader never decodes
    # it. Explicitly requesting that model should still report the loading error.
    HDF5.h5open(filename, "r+") do fid
        HDF5.delete_object(fid, "history/model")
        fid["history/model"] = "invalid model encoding"
    end
    restored = load_sim_history(filename)
    @test isnothing(restored.model)
    Logs.close_log(restored.log)
    @test_throws Exception load_sim_history(filename; load_model = true)

end

@testset "Histories without logging retain their run metadata" begin

    # Disabling time-series logging should not prevent saving the run's status or model.
    # There are no disk-backed datasets to keep open when this file is loaded.
    history = small_history(; log = Logs.NullLogOptions())
    filename = joinpath(mktempdir(), "history.h5")
    save_sim_history(filename, history; save_model = true)
    restored = load_sim_history(filename; load_model = true)
    @test restored.log isa Logs.NullLog
    @test isempty(keys(restored))
    @test restored.stop == history.stop
    @test restored.model == history.model
    @test restored.t_stop == exact_time(Float64(history.t_stop))
    Logs.close_log(restored.log)

end

@testset "Termination records preserve meaning without live objects" begin

    # Simple built-in reasons retain their concrete types. Custom reasons and hooks hold
    # a running task here to demonstrate that saving never traverses their live contents.
    history = small_history()
    task = current_task()
    reasons = [
        SystemsOfSystems.UnknownStopReason(),
        ReachedEndTime(history.t_stop),
        ModelRequestedStop("/child", "Finished."),
        Interrupted(history.t_stop),
        Solvers.SolverFailedToConverge(0.5),
        Solvers.SolverStepSizeUnderflow(0.5, 1e-20),
        HookRequestedStop(history.t_stop, ResourceHook(task)),
        CustomStop(task),
        CustomFailure(task),
        RecordedStop("Custom.Stop", "A recorded stop.", "Extra stop details."),
        RecordedFailure("Custom.Failure", "A recorded failure.", "Extra failure details."),
    ]
    filename = joinpath(mktempdir(), "history.h5")
    for stop in reasons

        save_sim_history(filename, with_stop(history, stop))
        restored = load_sim_history(filename)
        @test describe(restored.stop) == describe(stop)
        @test succeeded(restored) == succeeded(with_stop(history, stop))
        if stop isa Union{HookRequestedStop, CustomStop, CustomFailure}
            expected_type = stop isa AbstractFailureReason ? RecordedFailure : RecordedStop
            @test restored.stop isa expected_type
            @test restored.stop.original_type == string(typeof(stop))
        else
            @test typeof(restored.stop) == typeof(stop)
            @test all(
                isequal(getfield(restored.stop, f), getfield(stop, f))
                for f in fieldnames(typeof(stop))
            )
        end
        Logs.close_log(restored.log)

    end

    # Exception diagnostics include source locations, but no compiler objects or live
    # exception payloads. Re-saving a record must retain its original identity and details.
    stop = try
        error("Expected saved failure.")
    catch err
        EncounteredError(0.5, err, stacktrace(catch_backtrace()))
    end
    save_sim_history(filename, with_stop(history, stop))
    restored = load_sim_history(filename)
    @test restored.stop isa RecordedFailure
    @test !succeeded(restored)
    @test occursin("Expected saved failure.", restored.stop.details)
    @test occursin("test_sim_history_persistence.jl", restored.stop.details)

    # Saving a restored failure must retain the original exception's name and diagnostics,
    # both in its Julia record and in the datasets available to other HDF5 readers.
    copied = joinpath(mktempdir(), "copied.h5")
    save_sim_history(copied, restored)
    again = load_sim_history(copied)
    @test again.stop.original_type == restored.stop.original_type
    @test again.stop.details == restored.stop.details
    HDF5.h5open(copied, "r") do fid
        @test read(fid["history/stop/type"]) == string(typeof(stop))
        @test read(fid["history/stop/details"]) == restored.stop.details
    end
    Logs.close_log(again.log)
    Logs.close_log(restored.log)

end

@testset "Existing HDF5 logs can become history files without rewriting the log" begin

    # Direct logging creates /log before any history metadata exists. Root-model samples
    # must be inside that group too, so copying the group includes the complete log.
    filename = joinpath(mktempdir(), "live.h5")
    history = small_history(; log = Logs.HDF5LogOptions(filename))
    @test HDF5.name(history.log.group) == "/log"
    @test !haskey(history.log.fid, "history")
    @test haskey(history.log.group, "timeseries/x")
    @test !haskey(history.log.fid, "timeseries")

    # Saving elsewhere should copy both samples and additional metadata without changing
    # the source log. Include an attribute to cover metadata outside the child datasets.
    expected = history["/child"]["y"].data[:]
    history.log.group["extra_log_metadata"] = "Preserved when saving history."
    HDF5.attributes(history.log.group)["source"] = "Direct simulation"
    copied = joinpath(mktempdir(), "copied.h5")
    save_sim_history(copied, history)
    restored = load_sim_history(copied)
    @test restored["/child"]["y"].data[:] == expected
    @test read(HDF5.attributes(restored.log.group)["source"]) == "Direct simulation"
    Logs.close_log(restored.log)

    # Saving to the original file adds metadata beside the existing log. A second save
    # replaces that metadata while keeping the original sample handles usable.
    save_sim_history(filename, history; save_model = true)
    @test HDF5.name(history.log.group) == "/log"
    @test read(history.log.group["extra_log_metadata"]) == "Preserved when saving history."
    @test history["/child"]["y"].data[:] == expected
    save_sim_history(filename, history)
    @test history["/child"]["y"].data[:] == expected
    Logs.close_log(history.log)

    # The second save omitted the model, so the earlier model entry must be gone. This
    # loaded log is read-only and cannot be used to save back into its own file.
    restored = load_sim_history(filename; load_model = true)
    @test isnothing(restored.model)
    @test restored["/child"]["y"].data[:] == expected
    @test_throws ErrorException save_sim_history(filename, restored)
    Logs.close_log(restored.log)

    # Existing callers can still load just the log, including from a whole-history file.
    log, root = Logs.load_hdf5_log(filename)
    @test root.models.child["y"].data[:] == expected
    Logs.close_log(log)

end

@testset "Custom paths and groups share a file without taking ownership" begin

    # A results container can hold independent runs and unrelated application data. A
    # history records the log's path, so loading only requires locating the history group.
    history = small_history()
    filename = joinpath(mktempdir(), "runs.h5")
    HDF5.h5open(filename, "w") do fid

        fid["experiment"] = "Two runs"
        save_sim_history(fid, "/runs/one/history", history; log_path = "/runs/one/log")
        hg = HDF5.create_group(fid, "/runs/two/metadata")
        lg = HDF5.create_group(fid, "/runs/two/samples")
        save_sim_history(hg, history; log_group = lg, save_model = true)
        @test read(hg["log_path"]) == "/runs/two/samples"
        @test read(fid["runs/one/history/log_path"]) == "/runs/one/log"

        # Loading through a caller's group borrows the file. Closing the returned log must
        # leave all of the caller's handles valid for reading and writing the other runs.
        loaded = load_sim_history(hg; load_model = true)
        @test loaded["/child"]["y"].data[:] == history["/child"]["y"].data
        Logs.close_log(loaded.log)
        @test isopen(fid)
        @test isvalid(hg)
        @test isvalid(lg)

        # Replacing a history removes old optional fields, while other runs and metadata
        # remain intact. Group loaders and closers never take the caller's file ownership.
        save_sim_history(hg, history; log_group = lg)
        @test !haskey(hg, "model")
        loaded = load_sim_history(fid, "/runs/one/history")
        @test loaded.stop == history.stop
        Logs.close_log(loaded.log)
        @test read(fid["experiment"]) == "Two runs"

        # A null log has no retained handles, but its loader must still leave the caller
        # in charge of the container holding other histories.
        quiet = small_history(; log = Logs.NullLogOptions())
        save_sim_history(fid, "/quiet/history", quiet; log_path = "/quiet/log")
        @test load_sim_history(fid, "/quiet/history").log isa Logs.NullLog
        @test isopen(fid)

        # Loading just the log follows the same ownership rule as loading a whole history.
        log, root = Logs.load_hdf5_log(lg)
        @test root["x"].data[:] == history["/"]["x"].data
        Logs.close_log(log)
        @test isvalid(lg)

        # Overlapping destinations cannot be replaced safely; validation must precede
        # deletion of either history metadata or the source log's datasets.
        @test_throws ArgumentError save_sim_history(hg, history; log_group = hg)
        @test_throws ArgumentError save_sim_history(fid["runs/two"], history; log_group = lg)
        @test read(hg["log_path"]) == "/runs/two/samples"
        close(hg)
        close(lg)

    end
    loaded = load_sim_history(filename; history_path = "/runs/two/metadata")
    @test loaded["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(loaded.log)

    # A direct-to-disk simulation can select its own log group before history exists.
    filename = joinpath(mktempdir(), "custom.h5")
    direct = small_history(; log = Logs.HDF5LogOptions(; filename, path = "/samples"))
    @test HDF5.name(direct.log.group) == "/samples"
    save_sim_history(filename, direct; history_path = "/run")
    @test read(direct.log.fid["run/log_path"]) == "/samples"
    @test_throws ArgumentError save_sim_history(filename, direct; history_path = "/samples")
    @test direct["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(direct.log)
    loaded = load_sim_history(filename; history_path = "/run")
    @test loaded.stop == history.stop
    Logs.close_log(loaded.log)
    log, _ = Logs.load_hdf5_log(filename; path = "/samples")
    @test log["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(log)

    # Filename convenience methods also support custom locations for in-memory logs.
    filename = joinpath(mktempdir(), "custom_copy.h5")
    save_sim_history(filename, history; history_path = "/run", log_path = "/samples")
    loaded = load_sim_history(filename; history_path = "/run")
    @test loaded.stop == history.stop
    Logs.close_log(loaded.log)
    Logs.save_log_to_hdf5(filename, history.log; path = "/samples")
    log, _ = Logs.load_hdf5_log(filename; path = "/samples")
    @test log["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(log)

end

@testset "History do blocks close their files on success and failure" begin

    # The callback can read lazy samples while the file is open. Its copied result remains
    # usable afterward, and loading keywords must reach the regular filename loader.
    history = small_history()
    filename = joinpath(mktempdir(), "history.h5")
    save_sim_history(filename, history; history_path = "/run", save_model = true)
    file = Ref{HDF5.File}()
    samples = load_sim_history(filename; history_path = "/run", load_model = true) do loaded
        file[] = loaded.log.fid
        @test isopen(file[])
        @test loaded.model == history.model
        collect(loaded["/"]["x"].data)
    end
    @test !isopen(file[])
    @test samples == history["/"]["x"].data

    # Retain the actual file handle to verify immediate closure without relying on garbage
    # collection. Cleanup must not swallow or replace an exception from the callback.
    failure = ErrorException("Expected history callback failure.")
    caught = try
        load_sim_history(filename; history_path = "/run") do loaded
            file[] = loaded.log.fid
            throw(failure)
        end
    catch err
        err
    end
    @test caught === failure
    @test !isopen(file[])

    # Histories without logs have no retained file, but support the same callback interface.
    quiet = small_history(; log = Logs.NullLogOptions())
    save_sim_history(filename, quiet)
    result = load_sim_history(filename) do loaded
        @test loaded.log isa Logs.NullLog
        loaded.stop
    end
    @test result == quiet.stop

end

@testset "Legacy standalone logs remain readable" begin

    # Before /log became the default, standalone files stored model fields at the root.
    filename = joinpath(mktempdir(), "legacy.h5")
    history = small_history()
    Logs.save_log_to_hdf5(filename, history.log; path = "/")
    log, _ = Logs.load_hdf5_log(filename)
    @test log["/child"]["y"].data[:] == history["/child"]["y"].data
    Logs.close_log(log)

end

end # module TestSimHistoryPersistence
