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
    AutomaticGC <: AbstractGCPolicy

This GC management policy does not interact with Julia's garbage collection in any way and
is the default for `simulate`.
"""
mutable struct AutomaticGC <: AbstractGCPolicy end
initialize_gc!(::AutomaticGC) = nothing
step_gc!(::AutomaticGC) = nothing
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
    policy.prior_gc_state = GC.enable(false)
end
step_gc!(::NoGCInTheLoop) = nothing
terminate_gc!(policy::NoGCInTheLoop) = GC.enable(policy.prior_gc_state)

"""
    MinorGCInTheLoop <: AbstractGCPolicy

This GC management policy turns off automatic GC and runs a minor collection at the end of
every step of the simulation, allowing all temporary allocations from the step to be
immediately cleaned up. This is a good policy for simulations that need predictable
runtime on each step, such as simulations that interact with external real-time processes or
hardware.
"""
@kwdef mutable struct MinorGCInTheLoop <: AbstractGCPolicy
    prior_gc_state::Bool = false
    steps_per_gc::Int64 = 1
    steps_until_gc::Int64 = 0
end
function initialize_gc!(policy::MinorGCInTheLoop)
    GC.gc(true)
    policy.prior_gc_state = GC.enable(false)
    policy.steps_until_gc = policy.steps_per_gc
end
function step_gc!(::MinorGCInTheLoop)
    policy.steps_until_gc -= 1
    if policy.steps_until_gc <= 0
        GC.gc(false)
        policy.steps_until_gc = policy.steps_per_gc
    end
end
function terminate_gc!(policy::MinorGCInTheLoop)
    GC.enable(policy.prior_gc_state)
    GC.gc(true)
end

end
