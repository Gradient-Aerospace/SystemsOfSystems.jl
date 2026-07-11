module TestRandomVariableSeeds

using Test
using Random: Xoshiro, rand, randn
using SystemsOfSystems

first_randn(seed) = randn(Xoshiro(seed))

function described_randn(seed)
    return RandomVariableDescription{Float64}(
        (rng, t) -> randn(rng);
        seed = seed,
        title = "",
        dimensions = ["" => ""],
    )
end

function described_rand(seed)
    return RandomVariableDescription{Float64}(
        (rng, t) -> rand(rng);
        seed = seed,
        title = "",
        dimensions = ["" => ""],
    )
end

# Test that random variables without explicit `RandomVariableDescription` seeds still get
# their own default streams based on their field names.
@testset "default random variable seeds" begin

    seed = BranchingSeed(7, "")

    model_description = ModelDescription(;
        continuous_random_variables = (;
            w_draw = (rng, t_last, t_next) -> randn(rng),
        ),
        discrete_random_variables = (;
            x_draw = (rng, t) -> randn(rng),
            y_draw = (rng, t) -> randn(rng),
        ),
    )

    model = initialize(model_description; seed = seed)

    # The default seed for each random variable should be the model's seed branched by that
    # variable's own field name.
    @test model.w_draw == first_randn(seed / "w_draw")
    @test model.x_draw == first_randn(seed / "x_draw")
    @test model.y_draw == first_randn(seed / "y_draw")

    # Two same-shaped random variables in the same model should not be locked to identical
    # streams.
    @test model.x_draw != model.y_draw

end

@testset "adding random variables preserves existing streams" begin

    seed = BranchingSeed(9, "")

    base_model_description = ModelDescription(;
        discrete_random_variables = (;
            x_draw = (rng, t) -> randn(rng),
            y_draw = (rng, t) -> randn(rng),
        ),
    )

    expanded_model_description = ModelDescription(;
        discrete_random_variables = (;
            x_draw = (rng, t) -> randn(rng),
            added_draw = (rng, t) -> randn(rng),
            y_draw = (rng, t) -> randn(rng),
        ),
    )

    base_model = initialize(base_model_description; seed = seed)
    expanded_model = initialize(expanded_model_description; seed = seed)

    # Adding a new random variable should only create a new stream. Existing variables keep
    # their field-name-derived streams regardless of named-tuple order.
    @test expanded_model.added_draw == first_randn(seed / "added_draw")
    @test expanded_model.x_draw == base_model.x_draw
    @test expanded_model.y_draw == base_model.y_draw

end

@testset "explicit seeds override model path" begin

    seed = BranchingSeed(13, "")
    explicit_seed = seed / "shared_explicit_draw"

    submodel_description = ModelDescription(;
        discrete_random_variables = (;
            explicit_draw = described_randn(explicit_seed),
            default_draw = (rng, t) -> randn(rng),
        ),
    )

    model_a = initialize(
        ModelDescription(; models = (; left = submodel_description, ));
        seed = seed,
    )
    model_b = initialize(
        ModelDescription(; models = (; right = submodel_description, ));
        seed = seed,
    )

    # The explicit seed should reproduce the same draw even if the model description is
    # mounted at a different path.
    @test model_a.left.explicit_draw == first_randn(explicit_seed)
    @test model_b.right.explicit_draw == first_randn(explicit_seed)

    # Bare random variables should still follow their recursive model path.
    @test model_a.left.default_draw == first_randn(seed / "left" / "default_draw")
    @test model_b.right.default_draw == first_randn(seed / "right" / "default_draw")
    @test model_a.left.default_draw != model_b.right.default_draw

end

@testset "white noise variables use field seeds" begin

    seed = BranchingSeed(17, "")

    model_description = ModelDescription(;
        continuous_random_variables = (;
            nu_x = ContinuousWhiteNoise(1.0),
            nu_y = ContinuousWhiteNoise(1.0),
        ),
        discrete_random_variables = (;
            eta_x = DiscreteWhiteNoise(1.0),
            eta_y = DiscreteWhiteNoise(1.0),
        ),
    )

    model = initialize(model_description; seed = seed)

    # The white-noise helpers should get the same default field-name streams as generic
    # random-variable functions.
    @test model.nu_x == first_randn(seed / "nu_x")
    @test model.nu_y == first_randn(seed / "nu_y")
    @test model.eta_x == first_randn(seed / "eta_x")
    @test model.eta_y == first_randn(seed / "eta_y")
    @test model.nu_x != model.nu_y
    @test model.eta_x != model.eta_y

end

@testset "nested model default seeds" begin

    seed = BranchingSeed(21, "")

    leaf_description = ModelDescription(;
        continuous_random_variables = (;
            w_draw = (rng, t_last, t_next) -> randn(rng),
        ),
        discrete_random_variables = (;
            x_draw = (rng, t) -> randn(rng),
        ),
    )

    model_description = ModelDescription(;
        discrete_random_variables = (;
            root_draw = (rng, t) -> randn(rng),
        ),
        models = (;
            outer = ModelDescription(;
                discrete_random_variables = (;
                    outer_draw = (rng, t) -> randn(rng),
                ),
                models = (;
                    inner = leaf_description,
                ),
            ),
        ),
    )

    model = initialize(model_description; seed = seed)

    # Default seeds should include the full recursive model path before the variable name.
    @test model.root_draw == first_randn(seed / "root_draw")
    @test model.outer.outer_draw == first_randn(seed / "outer" / "outer_draw")
    @test model.outer.inner.w_draw == first_randn(seed / "outer" / "inner" / "w_draw")
    @test model.outer.inner.x_draw == first_randn(seed / "outer" / "inner" / "x_draw")

end

# Test that random draws taken during `initialize` match the draws from `simulate`. Note
# that this only works for submodel descriptions when random variables record their own
# seeds with `RandomVariableDescription`.
@testset "initialize matches simulate with explicit seeds" begin

    function init_fcn(t, _, seed)

        # Let the submodel initialize with a derived seed, as a parent model normally would.
        submodel_seed = seed / "submodel"

        submodel_init = ModelDescription(;
            continuous_outputs = (;
                z = 0.,
            ),
            discrete_random_variables = (;
                z_draw = described_randn(submodel_seed / "z_draw"),
            ),
        )

        # Use that initialization. This should take the same draw as the full simulation's
        # submodel will take.
        submodel = initialize(submodel_init)

        return ModelDescription(;
            continuous_outputs = (;
                x = 0., # Record the discrete draws on the first continuous sample.
                y = 0.,
            ),
            discrete_states = (;
                z = submodel.z_draw, # Store the submodel's initial draw so we can test it.
            ),
            discrete_random_variables = (;
                x_draw = described_randn(seed / "x_draw"),
                y_draw = described_rand(seed / "y_draw"),
            ),
            models = (;
                submodel = submodel_init,
            ),
        )

    end

    # Use both initialize and simulate to see if the same draws are happening.
    model = initialize(nothing; init_fcn = init_fcn, seed = 5)
    history, _, _ = simulate(
        nothing;
        init_fcn = init_fcn,
        rates_fcn = (t, model) -> RatesOutput(;
            outputs = (;
                x = model.x_draw,
                y = model.y_draw,
            ),
            models = (;
                submodel = RatesOutput(;
                    outputs = (;
                        z = model.submodel.z_draw,
                    ),
                ),
            ),
        ),
        t = 0 : 1 : 10,
        seed = 5,
    )

    # The initialized model should match the first samples in the time histories for the
    # outputs that record the draws.
    @test model.x_draw == history["/"]["x"].data[1]
    @test model.y_draw == history["/"]["y"].data[1]
    @test model.z == history["/submodel"]["z"].data[1]

end

end
