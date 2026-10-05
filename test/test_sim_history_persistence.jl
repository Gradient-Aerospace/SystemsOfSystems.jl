module TestSimHistoryPersistence

using Test
using SystemsOfSystems
using SystemsOfSystems: Logs, Hooks, exact_time,
    ReachedEndTime, HookRequestedStop, Interrupted, EncounteredError,
    TerminationSummary, AbstractTerminationReason,
    finished, failed, interrupted, describe, details,
    AbstractCleanupError, CleanupError, CleanupErrorSummary, cleanup_context, cleanup_details
import HDF5
import HDF5Vectors

# Use both a root model and a child to exercise relative storage paths. Fractional start
# and stop times also expose the intended conversion through Float64 in the saved record.
function small_history(; log = Logs.BasicLogOptions(), x = 1.0)
    return simulate(
        nothing;
        t = (1//3, 2//3),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x),
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

function termination_api_values_are_equal(a, b)
    return (
        finished(a) == finished(b) &&
        failed(a) == failed(b) &&
        interrupted(a) == interrupted(b) &&
        describe(a) == describe(b) &&
        details(a) == details(b)
    )
end

# This custom error has no context or details fields. Persistence must use its API and
# must not traverse the live task stored in its payload.
struct CustomCleanupError <: AbstractCleanupError
    payload::Task
end
SystemsOfSystems.cleanup_context(::CustomCleanupError) = "resource /output"
SystemsOfSystems.cleanup_details(::CustomCleanupError) = "Flush failed."

@testset "Cleanup summaries preserve diagnostics and order through re-saving" begin

    original = small_history(; log = Logs.NullLogOptions())
    cleanup_errors = AbstractCleanupError[
        CleanupError(
            "hook TestHook", ErrorException("Hook close failed"), stacktrace(),
        ),
        CustomCleanupError(current_task()),
    ]
    expected = [(string(typeof(err)), cleanup_context(err), cleanup_details(err))
        for err in cleanup_errors]
    history = SimHistory(
        original.t_start, original.t_stop, original.log, original.model, original.stop,
        cleanup_errors,
    )
    filename = joinpath(mktempdir(), "cleanup.h5")
    save_sim_history(filename, history)

    HDF5.h5open(filename, "r") do file
        @test read(file["history/cleanup_errors/count"]) == 2
        for (index, values) in enumerate(expected)
            entry = file["history/cleanup_errors/$index"]
            @test Tuple(read(entry[name]) for name in ("type", "context", "details")) == values
            close(entry)
        end
    end
    load_sim_history(filename) do loaded

        # Check the whole ordered diagnostic record, including the original type labels,
        # both on loading and after saving the loaded summaries again.
        @test succeeded(loaded)
        @test all(err -> err isa CleanupErrorSummary, loaded.cleanup_errors)
        @test [(err.type, cleanup_context(err), cleanup_details(err))
            for err in loaded.cleanup_errors] == expected

        copied = joinpath(mktempdir(), "copied.h5")
        save_sim_history(copied, loaded)
        load_sim_history(copied) do reloaded
            @test [(err.type, cleanup_context(err), cleanup_details(err))
                for err in reloaded.cleanup_errors] == expected
        end

    end

    # Type labels are optional provenance for older entries. Actual context and diagnostic
    # text are required; a partial entry must fail rather than invent those API results.
    HDF5.h5open(filename, "r+") do file

        entry = file["history/cleanup_errors/1"]
        HDF5.delete_object(entry, "type")
        loaded = load_sim_history(file, "history")
        @test first(loaded.cleanup_errors).type == ""
        @test cleanup_details(first(loaded.cleanup_errors)) == expected[1][3]
        for name in ("context", "details")

            value = read(entry[name])
            HDF5.delete_object(entry, name)
            @test_throws KeyError load_sim_history(file, "history")
            entry[name] = value

        end
        close(entry)

    end

    # Files without the optional cleanup group still represent an empty error list.
    HDF5.h5open(filename, "r+") do file
        HDF5.delete_object(file["history"], "cleanup_errors")
    end
    load_sim_history(filename) do loaded
        @test isempty(loaded.cleanup_errors)
    end

end

# A hook and a custom reason retain a running task. Saving should evaluate the termination
# API without attempting to serialize that task.
struct ResourceHook <: Hooks.AbstractHook
    task::Task
end
Base.show(io::IO, ::ResourceHook) = print(io, "ResourceHook")

struct CustomFailure <: AbstractTerminationReason
    task::Task
end
SystemsOfSystems.failed(::CustomFailure) = true
SystemsOfSystems.describe(::CustomFailure) = "A custom failure."
SystemsOfSystems.details(::CustomFailure) = "Custom diagnostics."

# Failing during policy selection exercises cleanup after the HDF5 file and log group
# exist, but before initialization can return a log to the caller.
struct FailingLogPolicy <: SystemsOfSystems.LoggingPolicies.AbstractLoggingPolicy end
function SystemsOfSystems.LoggingPolicies.get_model_logging_policy(::FailingLogPolicy, path)
    error("Expected log initialization failure.")
end

@testset "History file layout, lazy logs, and optional models" begin

    # Saving an in-memory log separates metadata from samples. The model is opt-in,
    # and times and termination labels should be readable using only HDF5.
    history = small_history()
    @test isempty(history.cleanup_errors)
    filename = joinpath(mktempdir(), "history.h5")
    @test isnothing(save_sim_history(filename, history))
    HDF5.h5open(filename, "r") do fid
        @test Set(keys(fid)) == Set(["history", "log"])
        @test read(fid["history/log_path"]) == "/log"
        @test fid["history/sim_history_version"] isa HDF5.Dataset
        @test ndims(fid["history/sim_history_version"]) == 0
        @test isempty(keys(HDF5.attributes(fid["history"])))
        @test isempty(keys(HDF5.attributes(fid["log"])))
        @test read(fid["history/sim_history_version"]) == 1
        @test read(fid["log/log_format_version"]) == 1
        for group in (fid["history"], fid["log"])
            @test read(group["systems_of_systems_version"]) ==
                string(Base.pkgversion(SystemsOfSystems))
        end
        @test read(fid["history/t_start"]) === Float64(history.t_start)
        @test read(fid["history/t_stop"]) === Float64(history.t_stop)
        @test read(fid["history/stop/type"]) == string(typeof(history.stop))
        @test read(fid["history/stop/finished"])
        @test !read(fid["history/stop/failed"])
        @test !read(fid["history/stop/interrupted"])
        @test read(fid["history/stop/summary"]) == describe(history.stop)
        @test read(fid["history/stop/details"]) == details(history.stop)
        @test haskey(fid, "log/models/child")
        @test !haskey(fid, "history/model")
    end

    # Loading restores the recorded times and model hierarchy, but leaves sample vectors
    # on disk. Asking for an unsaved model should simply return nothing.
    restored = load_sim_history(filename; load_model = true)
    @test restored.t_start == exact_time(Float64(history.t_start))
    @test restored.t_stop == exact_time(Float64(history.t_stop))
    @test termination_api_values_are_equal(restored.stop, history.stop)
    @test isempty(restored.cleanup_errors)
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

@testset "Public HDF5 paths can be read without Julia reconstruction" begin

    # Direct logging and copying an in-memory log have separate writers. Both must expose
    # the documented model hierarchy and numeric samples to an ordinary HDF5 consumer.
    # Custom group locations ensure readers follow log_path rather than assume /log.
    for direct in (false, true)

        filename = joinpath(mktempdir(), "public.h5")
        options = direct ? Logs.HDF5LogOptions(; filename, path = "/samples") :
            Logs.BasicLogOptions()
        history = small_history(; log = options)
        root_series = history["/"]["x"]
        child_series = history["/child"]["y"]
        expected_time = collect(root_series.time)
        expected_root = collect(root_series.data)
        expected_child = collect(child_series.data)
        save_sim_history(filename, history; history_path = "/run", log_path = "/samples")
        Logs.close_log(history.log)

        HDF5.h5open(filename, "r") do file

            # Navigate using only the public hierarchy and metadata datasets.
            root = file[read(file["run/log_path"])]
            @test read(root["names/models"]) == ["child"]
            @test read(root["names/continuous_states"]) == ["x"]
            @test isempty(read(root["names/discrete_states"]))
            @test isempty(read(root["names/constants"]))
            @test read(root["models/child/names/continuous_states"]) == ["y"]

            # Read numeric samples directly through the documented paths, without using
            # the Julia log loader or reconstructing interpolators and model types.
            series = root["timeseries/x"]
            @test read(series["path"]) == "/x"
            @test read(series["time_label"]) == root_series.time_dimension.label
            @test read(series["time_units"]) == root_series.time_dimension.units
            @test read(series["labels"]) == [d.label for d in root_series.dimensions]
            @test read(series["units"]) == [d.units for d in root_series.dimensions]
            @test !read(series["discrete"])
            @test read(series["time/data/values"]) == expected_time
            @test read(series["data/data/values"]) == expected_root
            @test read(root["models/child/timeseries/y/data/data/values"]) == expected_child

        end

    end

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
    @test termination_api_values_are_equal(restored.stop, history.stop)
    @test restored.model == history.model
    @test restored.t_stop == exact_time(Float64(history.t_stop))
    Logs.close_log(restored.log)

end

@testset "Termination records preserve meaning without live objects" begin

    # Cover nominal completion, interruption, a hook holding a live object, and a custom
    # failure classified by a method rather than its supertype. The persistence interface
    # is shared by all reasons, so each built-in type need not have a separate round trip.
    history = small_history(; log = Logs.NullLogOptions())
    task = current_task()
    reasons = (
        (ReachedEndTime(history.t_stop), (true, false, false)),
        (Interrupted(history.t_stop), (false, false, true)),
        (HookRequestedStop(history.t_stop, ResourceHook(task)), (true, false, false)),
        (CustomFailure(task), (false, true, false)),
    )
    filename = joinpath(mktempdir(), "history.h5")
    for (stop, classification) in reasons

        save_sim_history(filename, with_stop(history, stop))
        load_sim_history(filename) do restored

            @test restored.stop isa TerminationSummary
            @test restored.stop.type == string(typeof(stop))
            @test (finished(restored.stop), failed(restored.stop),
                interrupted(restored.stop)) == classification
            for f in (finished, failed, interrupted, describe, details)
                @test f(restored.stop) == f(stop)
            end
            @test succeeded(restored) == !classification[2]

        end

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
    @test restored.stop isa TerminationSummary
    @test !succeeded(restored)
    @test occursin("Expected saved failure.", restored.stop.details)
    @test occursin("test_sim_history_persistence.jl", restored.stop.details)

    # Saving a restored failure must retain the original exception's name and diagnostics,
    # both in its Julia record and in the datasets available to other HDF5 readers.
    copied = joinpath(mktempdir(), "copied.h5")
    save_sim_history(copied, restored)
    again = load_sim_history(copied)
    @test again.stop.type == restored.stop.type == string(typeof(stop))
    @test termination_api_values_are_equal(again.stop, restored.stop)
    HDF5.h5open(copied, "r") do file
        @test read(file["history/stop/type"]) == string(typeof(stop))
    end
    Logs.close_log(again.log)
    Logs.close_log(restored.log)

end

@testset "Termination records require every saved API field" begin

    # A partial record must not invent classification or diagnostic values. In particular,
    # losing the failure flag must produce an error rather than turn failure into success.
    history = with_stop(small_history(; log = Logs.NullLogOptions()),
        CustomFailure(current_task()))
    filename = joinpath(mktempdir(), "incomplete.h5")
    save_sim_history(filename, history)
    HDF5.h5open(filename, "r+") do file

        group = file["history/stop"]
        for field in fieldnames(TerminationSummary)

            name = string(field)
            value = read(group[name])
            HDF5.delete_object(group, name)
            @test_throws KeyError load_sim_history(file, "history")
            group[name] = value

        end
        close(group)

    end

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
        @test termination_api_values_are_equal(loaded.stop, history.stop)
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
    @test termination_api_values_are_equal(loaded.stop, history.stop)
    Logs.close_log(loaded.log)
    log, _ = Logs.load_hdf5_log(filename; path = "/samples")
    @test log["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(log)

    # Filename convenience methods also support custom locations for in-memory logs.
    filename = joinpath(mktempdir(), "custom_copy.h5")
    save_sim_history(filename, history; history_path = "/run", log_path = "/samples")
    loaded = load_sim_history(filename; history_path = "/run")
    @test termination_api_values_are_equal(loaded.stop, history.stop)
    Logs.close_log(loaded.log)
    Logs.save_log_to_hdf5(filename, history.log; path = "/samples")
    log, _ = Logs.load_hdf5_log(filename; path = "/samples")
    @test log["/"]["x"].data[:] == history["/"]["x"].data
    Logs.close_log(log)

end

@testset "Parent defaults keep independent runs separate" begin

    # A supplied parent scopes both default destinations. Writing another run under a
    # different parent must not redirect the first history to the second run's samples.
    first = small_history(; x = 1.0)
    second = small_history(; x = 2.0)
    filename = joinpath(mktempdir(), "runs.h5")
    HDF5.h5open(filename, "w") do file

        for (name, history) in (("first", first), ("second", second))
            parent = HDF5.create_group(file, name)
            save_sim_history(parent, "history", history)
            @test read(parent["history/log_path"]) == "/$name/log"
            close(parent)
        end
        @test !haskey(file, "log")
        for (name, history) in (("first", first), ("second", second))
            loaded = load_sim_history(file, "$name/history")
            @test loaded["/"]["x"].data[:] == history["/"]["x"].data
            Logs.close_log(loaded.log)
        end

        # A file parent still places the default log at /log. Absolute paths supplied
        # with a group parent remain rooted at the file, rather than that group.
        save_sim_history(file, "history", first)
        @test read(file["history/log_path"]) == "/log"
        parent = file["first"]
        save_sim_history(parent, "other_history", first; log_path = "/other_log")
        @test read(parent["other_history/log_path"]) == "/other_log"
        close(parent)

    end

end

@testset "Aliases cannot cause a saved log to erase itself" begin

    # Both a log and its parent may have additional hard or soft links. Comparisons must
    # use object identity, including the objects along each group's ancestor path.
    filename = joinpath(mktempdir(), "aliases.h5")
    history = small_history(;
        log = Logs.HDF5LogOptions(; filename, path = "/run/log"),
    )
    file = history.log.fid
    expected = history["/"]["x"].data[:]
    HDF5.API.h5l_create_hard(file, "/run/log", file, "/log_alias",
        HDF5.API.H5P_DEFAULT, HDF5.API.H5P_DEFAULT)
    HDF5.API.h5l_create_hard(file, "/run", file, "/parent_alias",
        HDF5.API.H5P_DEFAULT, HDF5.API.H5P_DEFAULT)
    HDF5.API.h5l_create_soft("/run", file, "/soft_parent",
        HDF5.API.H5P_DEFAULT, HDF5.API.H5P_DEFAULT)

    # Saving the same log through another name is a no-op. Using that object or an
    # aliased ancestor as history storage must instead fail before deleting any data.
    alias = file["log_alias"]
    Logs.save_log_to_hdf5(alias, history.log)
    @test haskey(history.log.group, "names")
    @test history["/"]["x"].data[:] == expected
    @test_throws ArgumentError save_sim_history(alias, history; log_group = history.log.group)
    close(alias)
    for path in ("/parent_alias", "/soft_parent")

        parent = file[path]
        @test_throws ArgumentError save_sim_history(parent, history;
            log_group = history.log.group)
        @test_throws ArgumentError Logs.save_log_to_hdf5(parent, history.log)
        close(parent)

        # Also test the opposite direction: the source log was opened through an alias,
        # while its ancestor destination uses the original name.
        source, _ = Logs.load_hdf5_log(file, "$path/log")
        parent = file["run"]
        @test_throws ArgumentError Logs.save_log_to_hdf5(parent, source)
        Logs.close_log(source)
        close(parent)

    end

    save_sim_history(filename, history)
    Logs.close_log(history.log)
    load_sim_history(filename) do loaded
        @test loaded["/"]["x"].data[:] == expected
    end

end

@testset "Open-file identity survives working-directory changes" begin

    # Preserve the old positional constructor and exercise a relative filename opened
    # inside another directory. The later save uses the same file's absolute name.
    directory = mktempdir()
    policy = SystemsOfSystems.LoggingPolicies.AllPassLoggingPolicy()
    options = Logs.HDF5LogOptions("relative.h5", policy)
    @test options.logging_policy === policy
    @test options.path == "/log"
    history = cd(directory) do
        small_history(; log = options)
    end
    filename = joinpath(directory, "relative.h5")
    expected = history["/"]["x"].data[:]
    save_sim_history(filename, history)
    @test history["/"]["x"].data[:] == expected
    Logs.close_log(history.log)
    load_sim_history(filename) do loaded
        @test termination_api_values_are_equal(loaded.stop, history.stop)
    end

    # Caller-owned files can still retain relative filenames. Group and filename saves
    # must recognize their identity without reinterpreting that name in the current cwd.
    # A non-default close policy also checks that reopening for comparison respects the
    # caller's file-access properties.
    file, loaded = cd(directory) do
        file = HDF5.h5open("relative.h5", "r+"; fclose_degree = :weak)
        file, load_sim_history(file, "history")
    end
    save_sim_history(filename, loaded; history_path = "/another_history")
    @test read(file["another_history/log_path"]) == "/log"
    @test loaded["/"]["x"].data[:] == expected
    Logs.close_log(loaded.log)
    @test isopen(file)
    close(file)

end

@testset "Failed setup releases newly opened handles" begin

    # Opening the second destination can fail after the first has been opened. The
    # history wrapper must release that first handle while preserving the caller's file.
    history = small_history()
    filename = joinpath(mktempdir(), "failure.h5")
    HDF5.h5open(filename, "w") do file
        close(HDF5.create_group(file, "history"))
        file["blocked"] = 1
        before = HDF5.API.h5f_get_obj_count(file, HDF5.API.H5F_OBJ_GROUP)
        @test_throws Exception save_sim_history(file, "history", history;
            log_path = "blocked/child")
        @test HDF5.API.h5f_get_obj_count(file, HDF5.API.H5F_OBJ_GROUP) == before
        @test isopen(file)
    end

    # A policy error occurs after direct-log creation has opened its file. Check that no
    # new file handle remains immediately afterward, without requiring garbage collection.
    GC.gc()
    before = HDF5.API.h5f_get_obj_count(HDF5.API.H5F_OBJ_ALL, HDF5.API.H5F_OBJ_FILE)
    options = Logs.HDF5LogOptions(; filename, logging_policy = FailingLogPolicy())
    @test_throws "Expected log initialization failure" Logs.create_log(
        options, ModelDescription(), SystemsOfSystems.Dimension("time", "s"),
    )
    @test HDF5.API.h5f_get_obj_count(HDF5.API.H5F_OBJ_ALL, HDF5.API.H5F_OBJ_FILE) == before

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
    @test termination_api_values_are_equal(result, quiet.stop)

end

@testset "Format checks reject unsupported or malformed versions before decoding" begin

    # These groups deliberately contain no data. An unsupported version must be rejected
    # before a loader attempts to read fields or deserialize Julia objects. Exercise both
    # group and filename entry points, including versions on a disabled log.
    filename = joinpath(mktempdir(), "versions.h5")
    for (name, version_name, loader) in (
        ("history", "sim_history_version", load_sim_history),
        ("log", "log_format_version", Logs.load_hdf5_log),
    )

        HDF5.h5open(filename, "w") do file

            group = HDF5.create_group(file, name)
            for version in (0, 2)
                group[version_name] = version
                @test_throws r"Unsupported .*Supported format version: 1" loader(group)
                HDF5.delete_object(group, version_name)
            end
            for version in (true, 1.0, "1", [1])
                group[version_name] = version
                @test_throws r"expected a scalar integer" loader(group)
                HDF5.delete_object(group, version_name)
            end
            @test isopen(file)
            @test isvalid(group)

            # The NullLog shortcut cannot bypass validation. File-owning wrappers must
            # report the same version error as group loaders.
            group[version_name] = 2
            if name == "log"
                group["is_null"] = true
                @test_throws r"Unsupported log_format_version" loader(group)
            end
            close(group)

        end
        @test_throws r"Unsupported .*Supported format version: 1" loader(filename)

    end

    # History metadata has always carried a version. Its absence is an error rather than
    # a request to guess a layout, and the do-block callback must not run on failed loads.
    HDF5.h5open(filename, "w") do file
        group = HDF5.create_group(file, "history")
        @test_throws r"Missing sim_history_version" load_sim_history(group)
        close(group)
    end
    called = Ref(false)
    @test_throws r"Missing sim_history_version" load_sim_history(filename) do history
        called[] = true
    end
    @test !called[]

end

@testset "Format versions and provenance follow the data being saved" begin

    # A direct log is versioned before history metadata exists. Reusing it must preserve
    # its own writer's provenance, while newly written history metadata names this writer.
    filename = joinpath(mktempdir(), "direct.h5")
    history = small_history(; log = Logs.HDF5LogOptions(filename))
    @test read(history.log.group["log_format_version"]) == 1
    @test read(history.log.group["systems_of_systems_version"]) ==
        string(Base.pkgversion(SystemsOfSystems))
    HDF5.delete_object(history.log.group, "systems_of_systems_version")
    history.log.group["systems_of_systems_version"] = "0.0.0"
    save_sim_history(filename, history; history_path = "/run")
    @test read(history.log.group["systems_of_systems_version"]) == "0.0.0"
    @test read(history.log.fid["run"]["systems_of_systems_version"]) ==
        string(Base.pkgversion(SystemsOfSystems))
    copied = joinpath(mktempdir(), "copied.h5")
    save_sim_history(copied, history)
    Logs.close_log(history.log)

    # Package-version provenance is not a compatibility gate and may be absent in files
    # written before it was recorded. Unknown extra metadata is also harmless.
    HDF5.h5open(copied, "r+") do file
        @test read(file["log"]["log_format_version"]) == 1
        @test read(file["log"]["systems_of_systems_version"]) == "0.0.0"
        HDF5.delete_object(file["history"], "systems_of_systems_version")
        file["history/additional_metadata"] = "Ignored by this reader"
    end
    load_sim_history(copied) do restored
        @test termination_api_values_are_equal(restored.stop, history.stop)
    end

    # A history may point to an unsupported log even when its own metadata is supported.
    # The history loader must still use the log's independent version check.
    HDF5.h5open(copied, "r+") do file
        HDF5.delete_object(file["log"], "log_format_version")
        file["log/log_format_version"] = 2
    end
    @test_throws r"Unsupported log_format_version" load_sim_history(copied)

    # Disabled logs are newly written records too, and must carry the same format marker.
    quiet = small_history(; log = Logs.NullLogOptions())
    save_sim_history(copied, quiet)
    HDF5.h5open(copied, "r") do file
        @test read(file["log"]["log_format_version"]) == 1
    end
    load_sim_history(copied) do restored
        @test restored.log isa Logs.NullLog
    end

end

@testset "Legacy standalone logs remain readable without being relabeled" begin

    # Existing unversioned logs can be at the root or in a selected group. Remove the
    # newly introduced datasets to reproduce those layouts, including omitted metadata
    # for which the legacy loader already provides defaults.
    history = small_history()
    for path in ("/", "/log")

        filename = joinpath(mktempdir(), "legacy.h5")
        Logs.save_log_to_hdf5(filename, history.log; path)
        HDF5.h5open(filename, "r+") do file
            group = file[path]
            HDF5.delete_object(group, "log_format_version")
            HDF5.delete_object(group, "systems_of_systems_version")
            HDF5.delete_object(group["names"], "models")
            HDF5.delete_object(group["timeseries/x"], "serialized_interpolator")
            close(group)
        end
        log, _ = Logs.load_hdf5_log(filename)
        @test log["/child"]["y"].data[:] == history["/child"]["y"].data
        @test log["/"]["x"].interpolator isa SystemsOfSystems.LinearInterpolation

        # HDF5 copying preserves the actual representation; it is not a migration. The
        # new history has its own version, while its copied legacy log stays unversioned.
        copied = joinpath(mktempdir(), "copied.h5")
        legacy = SimHistory(history.t_start, history.t_stop, log, nothing, history.stop)
        save_sim_history(copied, legacy)
        Logs.close_log(log)
        HDF5.h5open(copied, "r") do file
            @test read(file["history"]["sim_history_version"]) == 1
            @test !haskey(file["log"], "log_format_version")
            @test !haskey(file["log"], "systems_of_systems_version")
        end
        load_sim_history(copied) do restored
            @test restored["/child"]["y"].data[:] == history["/child"]["y"].data
        end

    end

end

end # module TestSimHistoryPersistence
