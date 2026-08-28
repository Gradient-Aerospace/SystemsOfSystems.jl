module TestComplexStates

using Test
using SystemsOfSystems

# Test that we can have a state that's a vector that changes size on every sample.
@kwdef struct MyModel
    buffer::Vector{Any}
end

function init(t, _, seed)
    return ModelDescription(;
        discrete_states = (;
            buffer = VariableDescription{Vector{Any}}(
                Any["Initial time = $(float(t))"];
                title = "Buffer",
                dimensions = [], # This state has no clear dimensionality.
                groups = [], # Because it has no dimension, it has no groups of dimensions.
            )
        ),
    )
end

# We'll grow the buffer on every sample. Note that we don't actually mutate, of course.
function updates(t, model)
    return UpdatesOutput(;
        updates = (;
            buffer = vcat(model.buffer, "Time = $(float(t))"),
        ),
    )
end

@testset "complex states" begin

    history = simulate(nothing; init_fcn = init, updates_fcn = updates, t = 1:1:10)

    @test length(history["/"]["buffer"].data) == 10
    for k in 1:10
        @test length(history["/"]["buffer"].data[k]) == k
        for (i, s) in enumerate(history["/"]["buffer"].data[k])
            if i == 1
                @test s == "Initial time = 1.0"
            else
                @test s == "Time = $(float(i))"
            end
        end
    end

end

end
