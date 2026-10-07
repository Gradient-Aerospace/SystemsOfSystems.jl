module GCPolicies

export AutomaticGC, NoGCInTheLoop, MinorGCInTheLoop

"""
    AbstractGCPolicy

An abstract type for garbage-collection management policies. Each subtype must implement
`initialize_gc!`, `step_gc!`, and `terminate_gc!`.
"""
abstract type AbstractGCPolicy end

"""
    initialize_gc!(policy::AbstractGCPolicy)

Allows a garbage collection policy to take whatever steps are necessary immediately prior to
entering the main simulation loop.
"""
function initialize_gc! end

"""
    step_gc!(policy::AbstractGCPolicy)

Allows a garbage collection policy run major, minor, or no garbage collections in the loop.
"""
function step_gc! end

"""
    terminate_gc!(policy::AbstractGCPolicy)

Allows a garbage collection policy to return GC to a normal state.
"""
function terminate_gc! end

"""
    enable_auto_full_collection(status)

This *experimental* function calls Julia's C internals to enable/disable automatic full
garbage collection. Explicit full collections are still allowed. This is not part of Julia's
API and hence may break even with minor updates. It is provided because Julia has no API
for disabling automatic full collections while still allowing minor collections.
"""
function enable_auto_full_collection(status)
    return ccall(:jl_gc_enable_auto_full_collection, Cint, (Cint,), status)
end

"""
    AutomaticGC <: AbstractGCPolicy

This GC management policy does not interact with Julia's garbage collection in any way and
is the default for `simulate`.
"""
mutable struct AutomaticGC <: AbstractGCPolicy end
initialize_gc!(::AutomaticGC) = nothing
@inline step_gc!(::AutomaticGC) = nothing
terminate_gc!(::AutomaticGC) = nothing

"""
    NoGCInTheLoop <: AbstractGCPolicy

This GC management policy requests a full collection before the simulation loop (if GC was
enabled) and then turns off GC during the loop. It restores the prior GC state when the loop
ends. It is the fastest policy, but temporary allocations will build up, potentially leading
to a crash, so it should only be used where there are absolutely no allocations or for short
simulations.
"""
@kwdef mutable struct NoGCInTheLoop <: AbstractGCPolicy
    prior_gc_state::Union{Nothing, Bool} = nothing
end
function initialize_gc!(policy::NoGCInTheLoop)
    GC.gc(true)
    policy.prior_gc_state = GC.enable(false)
end
@inline step_gc!(::NoGCInTheLoop) = nothing
function terminate_gc!(policy::NoGCInTheLoop)
    if !isnothing(policy.prior_gc_state)
        GC.enable(policy.prior_gc_state)
        policy.prior_gc_state = nothing
    end
end

"""
    MinorGCInTheLoop(; steps_per_gc = 1) <: AbstractGCPolicy

This *experimental* GC management policy requests a full collection before the simulation
loop (if GC was enabled), suppresses automatic full collections during the loop, and
requests a minor collection every `steps_per_gc` samples. (A full collection examines all
objects to see if they're still used and is expensive, whereas a minor collection only
examines new objects. In the sim loop, allocations from `rates_fcn` and `updates_fcn` can
likely be cleaned up as a routine minor collection.) _Automatic_ minor collections remain
enabled if GC was enabled before simulation.

If `steps_per_gc` is negative, this policy will not force a minor collection, relying
instead on Julia's automatic minor collections.

This can be a good policy for simulations that need predictable runtime, such as simulations
that interact with external real-time processes or hardware. However, this policy is
experimental because it relies on non-public Julia GC behavior. It may cease to function in
future Julia versions. Further, not all allocations are minor ("new generation" in Julia's
GC parlance); allocations in the "old generation" may continue to grow, leading to
out-of-memory errors or poor paging performance. This policy should be used with caution and
only where absolutely necessary for timing requirements.
"""
mutable struct MinorGCInTheLoop <: AbstractGCPolicy
    steps_per_gc::Int64
    prior_gc_state::Union{Nothing, Int}
    steps_until_gc::Int64
end
function MinorGCInTheLoop(; steps_per_gc = 1)
    return MinorGCInTheLoop(steps_per_gc, nothing, steps_per_gc)
end
function initialize_gc!(policy::MinorGCInTheLoop)
    policy.prior_gc_state = enable_auto_full_collection(0)
    GC.gc(true)
    policy.steps_until_gc = policy.steps_per_gc
end
@inline function step_gc!(policy::MinorGCInTheLoop)
    if policy.steps_per_gc > 0
        policy.steps_until_gc -= 1
    end
    if policy.steps_until_gc == 0
        GC.gc(false)
        policy.steps_until_gc = policy.steps_per_gc
    end
end
function terminate_gc!(policy::MinorGCInTheLoop)
    if !isnothing(policy.prior_gc_state)
        enable_auto_full_collection(policy.prior_gc_state)
        policy.prior_gc_state = nothing
    end
end

end
