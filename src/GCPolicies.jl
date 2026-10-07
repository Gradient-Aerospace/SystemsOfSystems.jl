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
    set_full_collection_status(status)

This *experimental* function calls Julia's C internals to enable/disable full garbage
collection. This is not part of Julia's API and hence may break even with minor updates. It
is provided because it is extremely useful since Julia has no API for disabling full
collections while still allowing minor collections.
"""
function set_full_collection_status(status)
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

This GC management policy turns off automatic GC and never explicitly requests GC. It is the
fastest policy, but temporary allocations will build up, potentially leading to a crash, so
it should only be used where there are absolutely no allocations or for short simulations.
"""
@kwdef mutable struct NoGCInTheLoop <: AbstractGCPolicy
    prior_gc_state::Bool = false
end
function initialize_gc!(policy::NoGCInTheLoop)
    GC.gc(true)
    policy.prior_gc_state = GC.enable(false)
end
@inline step_gc!(::NoGCInTheLoop) = nothing
function terminate_gc!(policy::NoGCInTheLoop)
    GC.enable(policy.prior_gc_state)
end

"""
    MinorGCInTheLoop(; steps_per_gc = 1) <: AbstractGCPolicy

This GC *experimental* management policy runs a full garbage collection prior to beginning
the simulation loop and then disables automatic GC and directly runs a minor collection
every `steps_per_gc` samples, allowing all temporary allocations from recent steps to be
immediately cleaned up. This is a good policy for simulations that need predictable runtime,
such as simulations that interact with external real-time processes or hardware. However,
this policy is experimental because it relies on non-public Julia GC behavior. This policy
may cease to function in future Julia versions.
"""
mutable struct MinorGCInTheLoop <: AbstractGCPolicy
    steps_per_gc::Int64
    prior_gc_state::Int
    steps_until_gc::Int64
end
MinorGCInTheLoop(; steps_per_gc = 1) = MinorGCInTheLoop(steps_per_gc, 0, steps_per_gc)
function initialize_gc!(policy::MinorGCInTheLoop)
    GC.gc(true)
    policy.prior_gc_state = set_full_collection_status(0)
    policy.steps_until_gc = policy.steps_per_gc
end
@inline function step_gc!(policy::MinorGCInTheLoop)
    policy.steps_until_gc -= 1
    if policy.steps_until_gc <= 0
        GC.gc(false)
        policy.steps_until_gc = policy.steps_per_gc
    end
end
function terminate_gc!(policy::MinorGCInTheLoop)
    set_full_collection_status(policy.prior_gc_state)
end

end
