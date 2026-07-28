module BranchingSeeds

export BranchingSeed, branch

import Random

"""
    BranchingSeed

A seed that can form a tree of reproducible random processes tracing back to a single
top-level seed. Here's an example of creating a `BranchingSeed` and creating a random number
generator from it:

```
seed = BranchingSeed(0, "")
rng = Xoshiro(seed)
```

Here is an example of a function that takes in a seed and creates multiple RNGs from it:

```
function foo(seed)

    # Create a top-level branching seed.
    branching_seed = BranchingSeed(seed, "")

    # Model Process A.
    branching_seed_a = branch(branching_seed, "a")
    rng_a = Xoshiro(branching_seed_a)
    x = randn(rng_a, 100)

    # Model Process B.
    branching_seed_b = branch(branching_seed, "b")
    rng_b = Xoshiro(branching_seed_b)
    y = randn(rng_b, 200)

    ...

end
```

In this example, the draws from `rng_a` and `rng_b` are independent of each other, but they
both still change when the top-level seed changes. This allows a user to model separate
random processes, where changing how many random draws are used as part of "process a"
doesn't change the draws of "process b". It's a very useful pattern for making models with
submodels; each submodel can `branch` from its parent's seed according to that model's name.
Then, even if models are swapped for different models, the remaining models will still
generate the same random draws over time.
"""
struct BranchingSeed
    salt::Int64
    breadcrumbs::String
end

"""
    branch(seed::BranchingSeed, name::AbstractString)

Creates a new `BranchingSeed` from the given `seed` by appending the given `name`.
"""
function branch(seed::BranchingSeed, name::AbstractString)
    return BranchingSeed(seed.salt, seed.breadcrumbs * "/" * name)
end

"""
    seed::BranchingSeed / name::AbstractString

This is syntactic sugar for `branch(seed, name)`.
"""
function Base.:/(seed::BranchingSeed, name::AbstractString)
    return branch(seed, name)
end

"Creates a Xoshiro RNG from the given BranchingSeed."
Random.Xoshiro(seed::BranchingSeed) = Random.Xoshiro(seed.salt + hash(seed.breadcrumbs))

end
