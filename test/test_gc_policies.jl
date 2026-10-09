module TestGCPolicies

using Test
using SystemsOfSystems
using SystemsOfSystems: EncounteredError, Interrupted

# There are two independent controls here: whether GC can run at all, and whether Julia
# can choose a full collection. A minor-only policy changes the second control, whereas
# NoGCInTheLoop changes the first. We record both so that restoring one cannot hide a
# mistake in the other.

# Earlier Julia 1.12 releases have no automatic-full-collection control. We still test
# AutomaticGC and NoGCInTheLoop there, representing the unavailable setting as `nothing`.
const HAS_FULL_GC_CONTROL = GCPolicies.HAS_AUTO_GC_CONTROL
const FULL_GC_STATES = HAS_FULL_GC_CONTROL ? (true, false) : (nothing,)
const GCState = NamedTuple{(:enabled, :full), Tuple{Bool, Union{Nothing, Bool}}}

# Keep the test matrix limited to policies supported by the running Julia version. The
# unsupported minor-policy case has its own test below, rather than skipping all GC tests.
const MANAGED_POLICIES = HAS_FULL_GC_CONTROL ?
    (GCPolicies.NoGCInTheLoop(), GCPolicies.MinorGCInTheLoop(; steps_per_gc = 8)) :
    (GCPolicies.NoGCInTheLoop(),)

"""
    gc_state()

Reads both GC settings without leaving either changed. `GC.enable` returns the previous
state, so briefly disabling GC lets us read that state and then restore it. If GC was
already disabled, this never enables it. Reading the full-collection setting requires the
C interface because Julia has no public getter for the experimental control we use. On
Julia versions without that control, `full` is `nothing`.
"""
function gc_state()

    enabled = GC.enable(false)
    GC.enable(enabled)
    full = if HAS_FULL_GC_CONTROL
        ccall(:jl_gc_auto_full_collection_is_enabled, Cint, ()) != 0
    else
        nothing
    end
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

        if HAS_FULL_GC_CONTROL
            GCPolicies.enable_auto_full_collection(Int(full))
        end
        GC.enable(enabled)
        return f()

    finally

        # The rest of the package suite must not inherit a failed test's GC settings.
        if HAS_FULL_GC_CONTROL
            GCPolicies.enable_auto_full_collection(Int(original.full))
        end
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
    # both. Each policy specification is reused across these cases. Every simulation must
    # create its own runtime, remembering that run's entry state.
    for policy in (GCPolicies.AutomaticGC(), MANAGED_POLICIES...),
        enabled in (true, false), full in FULL_GC_STATES, exception in (
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

    # An interval of one exercises collection at every sample; eight exercises skipped
    # samples and repeated collections. A negative interval exercises the new default:
    # automatic minor GC without forced collections. Restoration is covered above.

    # Creating a runtime must fail clearly on unsupported Julia versions. Constructing
    # the specification itself remains harmless. The normal simulation entry point must
    # reject it too, while leaving the caller's GC settings alone.
    if !HAS_FULL_GC_CONTROL

        policy = GCPolicies.MinorGCInTheLoop()
        original = gc_state()
        @test_throws ErrorException GCPolicies.create_gc_runtime(policy)
        @test_throws ErrorException policy_simulation(policy)
        @test gc_state() == original

    end

    for steps_per_gc in (HAS_FULL_GC_CONTROL ? (1, 8, -1) : ())

        with_gc_state(; enabled = true, full = true) do

            policy = GCPolicies.create_gc_runtime(GCPolicies.MinorGCInTheLoop(; steps_per_gc))
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
    # Explicitly request a collection at every step, since the default no longer does so.
    for specification in MANAGED_POLICIES

        with_gc_state(; enabled = false, full = first(FULL_GC_STATES)) do

            policy = GCPolicies.create_gc_runtime(
                specification isa GCPolicies.MinorGCInTheLoop ?
                GCPolicies.MinorGCInTheLoop(; steps_per_gc = 1) : specification,
            )

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

# The failure wrapper follows the same specification/runtime split as the real policies.
# Its runtime owns the real policy runtime so that cleanup exercises actual saved state.
struct InitializationFailureRuntime <: GCPolicies.AbstractGCPolicyRuntime
    policy::GCPolicies.AbstractGCPolicyRuntime
    stage::Symbol
end
function GCPolicies.create_gc_runtime(policy::InitializationFailure)
    return InitializationFailureRuntime(GCPolicies.create_gc_runtime(policy.policy), policy.stage)
end

# Before initialization, there is no saved setting to restore. After initialization,
# there is an active setting that must be restored even though the loop never started.
function GCPolicies.initialize_gc!(policy::InitializationFailureRuntime)
    if policy.stage === :after_initialization
        GCPolicies.initialize_gc!(policy.policy)
    end
    error("GC initialization failed")
end

# The wrapper always fails initialization, so reaching a sample would itself be a bug.
GCPolicies.step_gc!(::InitializationFailureRuntime) = error("Initialization did not succeed.")

# Delegate cleanup to the real policy so that its saved-state guard is part of the test.
GCPolicies.terminate_gc!(policy::InitializationFailureRuntime) =
    GCPolicies.terminate_gc!(policy.policy)

@testset "GC initialization failure cleanup" begin

    # Model failures above occur after successful initialization. Here, we specifically
    # need to check cleanup with and without a saved setting. An ordinary exception is
    # sufficient: interruption classification is already covered by the lifecycle test.
    for policy in MANAGED_POLICIES,
        stage in (:before_initialization, :after_initialization), full in FULL_GC_STATES

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

"""
    RecordingGCPolicy(events)

Records the GC lifecycle without changing GC settings. Direct policy tests cannot tell us
whether `simulate` actually creates a runtime and calls it once per sample. Sharing this
record with the model and resource callbacks lets us check the order of those calls too.
"""
struct RecordingGCPolicy <: GCPolicies.AbstractGCPolicy
    events::Vector{Symbol}
end

# Each run gets a new runtime, while retaining the test's shared event record. No actual
# collections are needed: this test checks integration, independent of Julia's GC timing.
struct RecordingGCRuntime <: GCPolicies.AbstractGCPolicyRuntime
    events::Vector{Symbol}
end
function GCPolicies.create_gc_runtime(policy::RecordingGCPolicy)
    push!(policy.events, :create)
    return RecordingGCRuntime(policy.events)
end
GCPolicies.initialize_gc!(runtime::RecordingGCRuntime) = push!(runtime.events, :initialize)
GCPolicies.step_gc!(runtime::RecordingGCRuntime) = push!(runtime.events, :gc_step)
GCPolicies.terminate_gc!(runtime::RecordingGCRuntime) = push!(runtime.events, :terminate)

"""
    RuntimeCreationFailure()

Rejects a policy before its runtime exists, just as an unsupported minor policy does.
This lets us test that rejection on every Julia version, including versions that support
minor-only GC. It must happen before model initialization can open resources or hooks.
"""
struct RuntimeCreationFailure <: GCPolicies.AbstractGCPolicy end
GCPolicies.create_gc_runtime(::RuntimeCreationFailure) = error("GC runtime creation failed")

@testset "simulation calls the GC runtime for each sample" begin

    # With no dynamics or extra model events, requested times 1 and 2 are the two loop
    # samples. The initial time is setup, so it must not add a third step_gc! call.
    events = Symbol[]
    history = simulate(
        nothing;
        t = (0, 1, 2),
        init_fcn = (args...) -> ModelDescription(;
            resources = (;
                observer = Resource(;
                    open_args = (),
                    open_fcn = args -> nothing,
                    close_fcn = resource -> push!(events, :close),
                ),
            ),
        ),
        updates_fcn = (t, model) -> begin
            push!(events, :update)
            nothing
        end,
        options = SimOptions(; gc_policy = RecordingGCPolicy(events), log = nothing),
    )

    # A terminal sample also receives a GC step. Cleanup must follow the final sample
    # and precede resource closure. This fails if the loop's step_gc! call is removed.
    @test succeeded(history)
    @test history.t_stop == 2
    @test events == [
        :create, :initialize, :update, :gc_step, :update, :gc_step, :terminate, :close,
    ]

    # Runtime creation can now reject an unsupported policy. A distinct error in the
    # model initializer exposes accidental initialization before that rejection; checking
    # the error text distinguishes the two paths without creating any real resources.
    original = gc_state()
    @test_throws r"GC runtime creation failed" simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> error("Model initialization should not run."),
        options = SimOptions(; gc_policy = RuntimeCreationFailure(), log = nothing),
    )
    @test gc_state() == original

end

@testset "nested simulations sharing options restore GC settings" begin

    # An immutable specification can be shared, but its saved entry state cannot be.
    # The inner run starts with the outer policy already active. Its cleanup must keep
    # that state active, and outer cleanup must then restore the original caller state.
    for policy in MANAGED_POLICIES, enabled in (true, false), full in FULL_GC_STATES

        with_gc_state(; enabled, full) do

            options = SimOptions(; gc_policy = policy, log = nothing)
            active = policy isa GCPolicies.NoGCInTheLoop ?
                (; enabled = false, full) : (; enabled, full = false)
            outer = simulate(
                nothing;
                t = (0, 1),
                init_fcn = (args...) -> ModelDescription(),
                updates_fcn = (t, model) -> begin

                    @test gc_state() == active
                    inner = simulate(
                        nothing;
                        t = (0, 1),
                        init_fcn = (args...) -> ModelDescription(),
                        options,
                    )
                    @test succeeded(inner)
                    @test gc_state() == active
                    return nothing

                end,
                options,
            )
            @test succeeded(outer)
            @test gc_state() == (; enabled, full)

        end

    end

end

end # TestGCPolicies
