module TestContinuousRandomVariables

using Test
using SystemsOfSystems

@kwdef struct ContinuousRandomWalk
    x::Float64 # Position so far
    nu::Float64 # # Noise draws
end

@testset "continuous draws retain small intervals at large epochs" begin

    # These exact times have the same Float64 representation, but their interval is a real
    # tenth of a second. Continuous random variables receive the exact start and the
    # solver's already-computed duration, so the interval cannot collapse to zero.
    t_start = 9_000_000_000_000_000//1
    t_stop = t_start + 1//10
    draw_inputs = Ref{Any}(nothing)
    noise = (rng, t_km1, dt_f) -> begin
        draw_inputs[] = (t_km1, dt_f)
        return ContinuousWhiteNoise(1.)(rng, t_km1, dt_f)
    end
    history = simulate(
        nothing;
        t = (t_start, t_stop),
        init_fcn = (args...) -> ModelDescription(;
            continuous_random_variables = (;
                noise,
            ),
        ),
        options = SimOptions(;
            solver = SystemsOfSystems.Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test isfinite(history.model.noise)
    @test draw_inputs[] == (t_start, 0.1)

end

# Here, we'll run a whole bunch of sims of a continuous-time random walk. The standard
# deviations of the end position is known as a function of the duration of the walk and the
# standard deviation of the noise. Note that we're using ContinuousWhiteNoise, which scales
# the random draws by 1/sqrt(dt) for us.
@testset "continuous_random_variables with $(typeof(solver))" for solver in (
    SystemsOfSystems.Solvers.Ralston2Options(; dt = 1//1),
    SystemsOfSystems.Solvers.RungeKutta4Options(; dt = 1//1),
    SystemsOfSystems.Solvers.DormandPrince54Options(),
)

    # Here are some arbitrary numbers.
    n           = 5000
    sigma_noise = 3.1
    t_final     = 20//1

    # Make an array of the final positions.
    final_position = [
        begin
            history = simulate(
                nothing;
                t = 0//1 : 1//10 : t_final, # Force some steps to happen (not one big step).
                seed,
                init_fcn = (args...) -> ModelDescription(;
                    type = ContinuousRandomWalk,
                    continuous_states = (;
                        x = 0.,
                    ),
                    continuous_random_variables = (;
                        nu = ContinuousWhiteNoise(sigma_noise),
                    ),
                ),
                rates_fcn = (t, model) -> RatesOutput(;
                    rates = (;
                        x = model.nu, # The derivative is just white noise.
                    ),
                ),
                options = SimOptions(; solver, ),
            )
            history["/"]["x"].data[end]
        end
        for seed in 1:n
    ]

    # See if the resulting positions matched theory. We could increase sims to increase the
    # resolution of this test, but this seems plenty sufficient to say whether continuous
    # random draws are handled properly by the solvers.
    sigma_position = sqrt(sum(final_position.^2) / n)
    sigma_expected = sigma_noise * sqrt(t_final)
    @test sigma_position ≈ sigma_expected rtol = 1e-2

end

end
