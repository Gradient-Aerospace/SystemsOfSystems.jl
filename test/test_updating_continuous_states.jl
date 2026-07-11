module TestUpdatingContinuousStates

using Test
using SystemsOfSystems

# Let's test that an object can fall for 1s but then resets back to its original position
# and velocity.
@testset "updating continuous states" begin

    history, _, _ = simulate(
        nothing;
        t = 0//1 : 1//2 : 5//1,
        init_fcn = (t, params, seed) -> ModelDescription(;
            continuous_states = (;
                x = 1.,
                x_dot = 0.,
            ),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (;
                x = model.x_dot,
                x_dot = -9.81,
            ),
        ),
        updates_fcn = (t, model) -> if is_regular_step_triggering(t, 1//1)
            UpdatesOutput(;
                updates = (;
                    x = 1.,
                    x_dot = 0.,
                ),
            )
        else
            UpdatesOutput()
        end,
    )

    # At the 1s intervals, the position always snaps back.
    for t in 0. : 1. : 5.
        @test history["/"]["x"](t) == 1.
        @test history["/"]["x_dot"](t) == 0.
    end

    # Off of those intervals, the position is changing as expected.
    for t in 0.5 : 1. : 4.5
        @test history["/"]["x"](t) ≈ 1. - 0.5 * 9.81 * 0.5^2 atol = 1e-6
        @test history["/"]["x_dot"](t) ≈ -9.81 * 0.5 atol = 1e-6
    end

    # Test interpolation, that the updated value isn't incorrectly interpolated backwards.
    # Note that this uses linear interpolation, so the velocity will be fine, but the
    # position will be off. It should be linearly interpolating between the value at t = 0.5
    # and t = 1.
    x_at_half = 1. - 0.5 * 9.81 * 0.5^2
    x_at_full = 1. - 0.5 * 9.81 * 1.0^2
    x_at_three_quarters = 0.5 * (x_at_half + x_at_full)
    for t in 0.75 : 1. : 4.75
        @test history["/"]["x"](t) ≈ x_at_three_quarters atol = 1e-6
        @test history["/"]["x_dot"](t) ≈ -9.81 * 0.75 atol = 1e-6
    end

end

end
