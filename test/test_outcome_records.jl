module TestOutcomeRecords

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers, describe, record_stop, record_cleanup_error,
    ReachedEndTime, ModelRequestedStop, Interrupted, EncounteredError,
    RecordedStop, RecordedFailure, AbstractStopReason, AbstractFailureReason,
    CleanupError, RecordedCleanupError

# These reasons deliberately retain a running task, which cannot be reconstructed as part
# of a saved run. Their records should use descriptions without retaining the task.
struct CustomStop <: AbstractStopReason
    task::Task
end
struct CustomFailure <: AbstractFailureReason
    task::Task
end

SystemsOfSystems.describe(::CustomStop) = "A custom stop."
SystemsOfSystems.describe(::CustomFailure) = "A custom failure."

@testset "Termination records preserve meaning without live objects" begin

    # Simple built-in reasons retain their concrete types and fields. These checks call
    # the public conversion directly, without saving or loading an HDF5 history.
    reasons = [
        SystemsOfSystems.UnknownStopReason(),
        ReachedEndTime(1//3),
        ModelRequestedStop("/child", "Finished."),
        Interrupted(1//3),
        Solvers.SolverFailedToConverge(0.5),
        Solvers.SolverStepSizeUnderflow(0.5, 1e-20),
        RecordedStop("Custom.Stop", "A recorded stop.", "Extra stop details."),
        RecordedFailure("Custom.Failure", "A recorded failure.", "Extra failure details."),
    ]
    for stop in reasons
        @test record_stop(stop) === stop
    end

    # Custom reasons keep their explanation and failure classification, but not the task.
    task = current_task()
    for stop in (CustomStop(task), CustomFailure(task))
        record = record_stop(stop)
        expected_type = stop isa AbstractFailureReason ? RecordedFailure : RecordedStop
        @test record isa expected_type
        @test record.original_type == string(typeof(stop))
        @test describe(record) == describe(stop)
        @test record_stop(record) === record
    end

    # Exception diagnostics include source locations, but no compiler objects or live
    # exception payloads. Converting a record again must retain its diagnostic text.
    exception, trace = try
        error("Expected record test failure")
    catch err
        (err, stacktrace(catch_backtrace()))
    end
    failure = record_stop(EncounteredError(0.5, exception, trace))
    @test failure isa RecordedFailure
    @test failure.details == sprint(showerror, exception, trace)
    @test record_stop(failure) === failure

    # Cleanup records use the same readable diagnostics and retain the resource context.
    cleanup = record_cleanup_error(CleanupError("resource /output", exception, trace))
    @test cleanup isa RecordedCleanupError
    @test cleanup.context == "resource /output"
    @test cleanup.details == failure.details
    @test record_cleanup_error(cleanup) === cleanup

end

end # module TestOutcomeRecords
