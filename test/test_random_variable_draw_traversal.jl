module TestRandomVariableDrawTraversal

using Test
using SystemsOfSystems

function initialization_artifacts(model_description)
    context = SystemsOfSystems.initialization_context(; seed = 31)
    return SystemsOfSystems.create_initialization_artifacts(model_description, context)
end

continuous_draw(rng, t_last, t_next) = t_last + t_next
discrete_draw(rng, t) = float(t)

@testset "random variable subtree metadata" begin

    model_description = ModelDescription(;
        models = (;
            continuous_branch = ModelDescription(;
                continuous_random_variables = (; continuous_draw, ),
                models = (;
                    unused_leaf = ModelDescription(),
                ),
            ),
            discrete_branch = ModelDescription(;
                models = (;
                    random_leaf = ModelDescription(;
                        discrete_random_variables = (; discrete_draw, ),
                    ),
                ),
            ),
            unused_branch = ModelDescription(;
                models = (;
                    unused_leaf = ModelDescription(),
                ),
            ),
        ),
    )

    artifacts = initialization_artifacts(model_description)
    ommd = artifacts.ommd

    # Each flag describes random variables anywhere below the model, excluding variables
    # directly on the model itself.
    @test ommd.models_have_continuous_random_variables
    @test ommd.models_have_discrete_random_variables
    @test !ommd.models.continuous_branch.models_have_continuous_random_variables
    @test !ommd.models.continuous_branch.models_have_discrete_random_variables
    @test !ommd.models.discrete_branch.models_have_continuous_random_variables
    @test ommd.models.discrete_branch.models_have_discrete_random_variables
    @test !ommd.models.unused_branch.models_have_continuous_random_variables
    @test !ommd.models.unused_branch.models_have_discrete_random_variables

end


@testset "random variable draws stop at irrelevant subtrees" begin

    model_description = ModelDescription(;
        models = (;
            continuous_branch = ModelDescription(;
                continuous_random_variables = (; continuous_draw, ),
                models = (;
                    unused_leaf = ModelDescription(),
                ),
            ),
            discrete_branch = ModelDescription(;
                models = (;
                    random_leaf = ModelDescription(;
                        discrete_random_variables = (; discrete_draw, ),
                    ),
                ),
            ),
            unused_branch = ModelDescription(;
                models = (;
                    unused_leaf = ModelDescription(),
                ),
            ),
        ),
    )

    artifacts = initialization_artifacts(model_description)
    ommd = artifacts.ommd
    initial_state = artifacts.msd

    continuous_state = SystemsOfSystems.draw_wc(1., 2., ommd, initial_state)

    # The continuous branch gets a new draw, but traversal ends there and preserves its
    # child states. Subtrees containing only discrete or no random variables are reused.
    @test continuous_state !== initial_state
    @test continuous_state.models.continuous_branch !==
        initial_state.models.continuous_branch
    @test continuous_state.models.continuous_branch.models ===
        initial_state.models.continuous_branch.models
    @test continuous_state.models.continuous_branch.continuous_random_variables ==
        (; continuous_draw = 3., )
    @test continuous_state.models.discrete_branch === initial_state.models.discrete_branch
    @test continuous_state.models.unused_branch === initial_state.models.unused_branch

    discrete_state = SystemsOfSystems.draw_wd(2, ommd, initial_state)

    # The discrete traversal follows the one relevant branch to its random leaf. The
    # continuous-only and entirely unused branches are reused without being reconstructed.
    @test discrete_state !== initial_state
    @test discrete_state.models.continuous_branch === initial_state.models.continuous_branch
    @test discrete_state.models.discrete_branch !== initial_state.models.discrete_branch
    @test discrete_state.models.discrete_branch.models.random_leaf !==
        initial_state.models.discrete_branch.models.random_leaf
    random_leaf = discrete_state.models.discrete_branch.models.random_leaf
    @test random_leaf.discrete_random_variables == (; discrete_draw = 2., )
    @test discrete_state.models.unused_branch === initial_state.models.unused_branch

end


@testset "random variable draws reuse an irrelevant root" begin

    artifacts = initialization_artifacts(ModelDescription(;
        models = (;
            branch = ModelDescription(;
                models = (;
                    leaf = ModelDescription(),
                ),
            ),
        ),
    ))

    # A hierarchy with no random variables should return immediately for either process.
    @test SystemsOfSystems.draw_wc(1., 2., artifacts.ommd, artifacts.msd) === artifacts.msd
    @test SystemsOfSystems.draw_wd(2, artifacts.ommd, artifacts.msd) === artifacts.msd

end

end # TestRandomVariableDrawTraversal
