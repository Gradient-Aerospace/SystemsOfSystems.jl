module TestGCPolicies

using Test
using SystemsOfSystems
using SystemsOfSystems: EncounteredError, Interrupted

# There are two independent controls here: whether GC can run at all, and whether Julia
# can choose a full collection. A minor-only policy changes the second control, whereas
# NoGCInTheLoop changes the first. We record both so that restoring one cannot hide a
# mistake in the other.
const GCState = NamedTuple{(:enabled, :full), Tuple{Bool, Bool}}

"""
    gc_state()

Reads both GC settings without leaving either changed. `GC.enable` returns the previous
state, so briefly disabling GC lets us read that state and then restore it. If GC was
already disabled, this never enables it. Reading the full-collection setting requires the
C interface because Julia has no public getter for the experimental control we use.
"""
function gc_state()
    enabled = GC.enable(false)
    GC.enable(enabled)
    full = ccall(:jl_gc_auto_full_collection_is_enabled, Cint, ()) != 0
    return (; enabled, full)
end

"""
    with_gc_state(f; enabled, full)

Runs a test with the supplied GC settings, then restores the settings that the test process
had on entry. The tests deliberately change process-wide behavior, so each needs a cleanup
path that works even if a policy throws or a test fails. This cleanup is independent of the
policy's own cleanup; assertions inside the block check the policy before we restore things.
"""
function with_gc_state(f; enabled, full)

    # Save the real entry state before making either change. A nested test can therefore
    # use this helper without assuming that the outer test left GC enabled.
    original = gc_state()
    try

        GCPolicies.enable_auto_full_collection(Int(full))
        GC.enable(enabled)
        return f()

    finally

        # The rest of the package suite must not inherit a failed test's GC settings.
        GCPolicies.enable_auto_full_collection(Int(original.full))
        GC.enable(original.enabled)

    end

end

"""
    policy_simulation(policy; exception = nothing)

Runs a small simulation and returns its history plus the GC settings observed during model
updates and resource closure. The model has no dynamics: we're testing the simulation's GC
lifecycle, rather than a solver. An optional exception is thrown at t = 1, after execution
has begun, so we can check failure and interruption cleanup without operating-system signals.

Resource closure is a useful observation point because it occurs after loop cleanup. Seeing
the original settings there proves that restoration happens before resources are closed,
rather than merely sometime before `simulate` returns.
"""
function policy_simulation(policy; exception = nothing)

    # Each run gets fresh observation records. The policy itself can be reused by the
    # calling test, which checks that it remembers this run's entry state.
    observed = GCState[]
    closed = GCState[]
    history = simulate(
        nothing;
        t = (0, 1, 2),
        init_fcn = (args...) -> ModelDescription(;
            resources = (;
                observer = Resource(;
                    open_args = (),
                    open_fcn = args -> nothing,
                    close_fcn = resource -> push!(closed, gc_state()),
                ),
            ),
        ),
        updates_fcn = (t, model) -> begin

            # Record the setting before throwing so that failed runs also tell us what
            # the model saw while the policy was active.
            push!(observed, gc_state())
            t == 1 && exception !== nothing && throw(exception)
            return nothing

        end,
        options = SimOptions(; gc_policy = policy, log = nothing),
    )
    return (; history, observed, closed)

end

"""
    measured_gc_call!(f, arg)

Calls `f(arg)` and reports how many collections and full collections occurred during that
call. Julia's internal `pause` counter counts collections, despite its name; it is not a
pause duration. These counters let us test collection requests without imposing a timing
limit or depending on when a particular object is reclaimed.

Only the call is measured. Test assertions and setup can allocate and trigger automatic GC,
so including them in the measurement could wrongly attribute a collection to the policy.
"""
function measured_gc_call!(f, arg)
    before = Base.gc_num()
    f(arg)
    after = Base.gc_num()
    return (;
        collections = after.pause - before.pause,
        full_collections = after.full_sweep - before.full_sweep,
    )
end

@testset "GC settings during simulation and after cleanup" begin

    # Both entry settings matter: a user may already have disabled all GC, full GC, or
    # both. Each policy instance is reused across these cases, so restoration must use
    # the current run's entry state rather than a state remembered from an earlier run.
    for policy in (
        GCPolicies.AutomaticGC(), GCPolicies.NoGCInTheLoop(),
        GCPolicies.MinorGCInTheLoop(; steps_per_gc = 8),
    ), enabled in (true, false), full in (true, false), exception in (
        nothing, ErrorException("GC policy test failure"), InterruptException(),
    )

        with_gc_state(; enabled, full) do

            # An ordinary model error should emit an error log. An interruption is a
            # supported stop request and should not be logged as an unexpected failure.
            result = if exception isa ErrorException
                @test_logs (:error,) policy_simulation(policy; exception)
            else
                @test_logs policy_simulation(policy; exception)
            end
            original = (; enabled, full)
            @test gc_state() == original
            @test result.closed == [original]

            # AutomaticGC leaves both controls alone. NoGCInTheLoop disables all GC;
            # MinorGCInTheLoop only suppresses automatic full GC. Neither enables GC
            # that was initially off.
            expected = if policy isa GCPolicies.NoGCInTheLoop
                (; enabled = false, full)
            elseif policy isa GCPolicies.MinorGCInTheLoop
                (; enabled, full = false)
            else
                original
            end
            @test !isempty(result.observed)
            @test all(==(expected), result.observed)

            # Restoration alone isn't enough: a broken policy could abort the simulation
            # and still restore its settings. Check that the intended exit actually occurred.
            if exception === nothing
                @test succeeded(result.history)
                @test result.history.t_stop == 2
            elseif exception isa InterruptException
                @test result.history.stop isa Interrupted
            else
                @test result.history.stop isa EncounteredError
                @test result.history.stop.exception === exception
            end

        end

    end

end

@testset "minor GC collection requests" begin

    # An interval of one exercises the default; eight exercises skipped samples and
    # repeated collections. A negative interval exercises automatic minor GC without
    # forcing any collections. The lifecycle test above already covers restoring flags.
    for steps_per_gc in (1, 8, -1)

        with_gc_state(; enabled = true, full = true) do

            policy = GCPolicies.MinorGCInTheLoop(; steps_per_gc)
            activity = measured_gc_call!(GCPolicies.initialize_gc!, policy)
            @test activity.full_collections >= 1

            # Two complete periods distinguish a working counter from a policy that
            # collects once but never resets. Negative intervals use the same sixteen
            # calls; none of those calls should force a collection.
            for sample in 1:16

                activity = measured_gc_call!(GCPolicies.step_gc!, policy)
                collection_due = steps_per_gc > 0 && sample % steps_per_gc == 0
                if collection_due
                    @test activity.collections >= 1
                else
                    @test activity.collections == 0
                end
                @test activity.full_collections == 0

            end

            # Suppressing automatic full GC must not block explicit full GC. This also
            # checks that initialization did not accidentally disable all GC instead.
            activity = measured_gc_call!(GC.gc, true)
            @test activity.full_collections >= 1
            GCPolicies.terminate_gc!(policy)

        end

    end

end

@testset "managed policies leave initially disabled GC off" begin

    # Observing disabled GC during a model update would miss a policy that briefly enabled
    # it to collect and disabled it again. Count collections around each lifecycle call
    # to establish that the caller's decision is respected throughout the entire run.
    for policy in (GCPolicies.NoGCInTheLoop(), GCPolicies.MinorGCInTheLoop())

        with_gc_state(; enabled = false, full = true) do

            for f in (
                GCPolicies.initialize_gc!, GCPolicies.step_gc!, GCPolicies.terminate_gc!,
            )

                activity = measured_gc_call!(f, policy)
                @test activity.collections == 0
                @test !gc_state().enabled

            end

        end

    end

end

"""
    InitializationFailure(policy, stage)

A test-only wrapper that throws before or after initializing a real managed policy. Real GC
initialization failures are difficult to cause reliably. This wrapper lets us exercise
the simulation's cleanup at both points without replacing the real
policy's restoration code. `stage` is `:before_initialization` or `:after_initialization`.
"""
struct InitializationFailure <: GCPolicies.AbstractGCPolicy
    policy::GCPolicies.AbstractGCPolicy
    stage::Symbol
end

# Before initialization, there is no saved setting to restore. After initialization,
# there is an active setting that must be restored even though the loop never started.
function GCPolicies.initialize_gc!(policy::InitializationFailure)
    if policy.stage === :after_initialization
        GCPolicies.initialize_gc!(policy.policy)
    end
    error("GC initialization failed")
end

# The wrapper always fails initialization, so reaching a sample would itself be a bug.
GCPolicies.step_gc!(::InitializationFailure) = error("Initialization did not succeed.")

# Delegate cleanup to the real policy so that its saved-state guard is part of the test.
GCPolicies.terminate_gc!(policy::InitializationFailure) =
    GCPolicies.terminate_gc!(policy.policy)

@testset "GC initialization failure cleanup" begin

    # Model failures above occur after successful initialization. Here, we specifically
    # need to check cleanup with and without a saved setting. An ordinary exception is
    # sufficient: interruption classification is already covered by the lifecycle test.
    for policy in (GCPolicies.NoGCInTheLoop(), GCPolicies.MinorGCInTheLoop()),
        stage in (:before_initialization, :after_initialization), full in (true, false)

        with_gc_state(; enabled = true, full) do

            # A before-initialization failure leaves the caller's settings alone; an
            # after-initialization failure must undo the policy's changes. In both cases
            # resources opened by simulation setup still need to be closed.
            failing = InitializationFailure(policy, stage)
            result = @test_logs (:error,) policy_simulation(failing)
            @test result.history.stop isa EncounteredError
            @test isempty(result.observed)
            @test result.closed == [(; enabled = true, full)]
            @test gc_state() == (; enabled = true, full)

        end

    end

end

end # TestGCPolicies
