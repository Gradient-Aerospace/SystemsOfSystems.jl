module TestCleanupErrors

using Test
using SystemsOfSystems: CleanupError, CleanupErrorSummary, cleanup_context, cleanup_details

@testset "Cleanup summaries retain the diagnostic API without persistence" begin

    # A live error retains its exception and trace. Its summary provides the same readable
    # context and diagnostics without needing HDF5 or retaining those process-local objects.
    err = try
        throw(ErrorException("Expected cleanup failure."))
    catch exception
        CleanupError("resource /output", exception, stacktrace(catch_backtrace()))
    end
    @test cleanup_context(err) == "resource /output"
    @test occursin("Expected cleanup failure.", cleanup_details(err))
    @test occursin("test_cleanup_errors.jl", cleanup_details(err))

    summary = CleanupErrorSummary(err)
    @test summary.type == string(typeof(err))
    @test cleanup_context(summary) == cleanup_context(err)
    @test cleanup_details(summary) == cleanup_details(err)
    @test CleanupErrorSummary(summary) === summary

end

end # module TestCleanupErrors
