module TestContinuousRandomVariables

using Test
using SystemsOfSystems

@kwdef struct ContinuousRandomWalk
    x::Float64 # Position so far
    nu::Float64 # # Noise draws
end

@testset "continuous draws retain small intervals at large epochs" begin

    # These exact times have the same Float64 representation, but their interval is a real
    # tenth of a second. Continuous random variables receive the exact start and a duration
    # computed from their schedule, so the interval cannot collapse to zero.
    t_start = 9_000_000_000_000_000//1
    t_stop = t_start + 1//10
    draw_inputs = Ref{Any}(nothing)
    noise = (rng, t_km1, dt_f) -> begin
        draw_inputs[] = (t_km1, dt_f)
        return ContinuousWhiteNoise(1., RegularSchedule(1//10))(rng, t_km1, dt_f)
    end
    history = simulate(
        nothing;
        t = (t_start, t_stop),
        init_fcn = (args...) -> ModelDescription(;
            continuous_random_variables = (;
                noise = ContinuousRandomVariable(
                    noise,
                    RegularSchedule(1//10),
                ),
            ),
        ),
        options = SimOptions(;
            solver = SystemsOfSystems.Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test isfinite(history.model.noise)
    @test draw_inputs[] == (t_start, 0.1)

end

@testset "continuous random-variable schedules are automatic" begin

    draw_inputs = Tuple{Rational{Int64}, Float64}[]
    draw = (rng, t_km1, dt_f) -> begin
        push!(draw_inputs, (t_km1, dt_f))
        return length(draw_inputs)
    end
    schedule = RegularSchedule(3//10)
    described_draw = RandomVariableDescription{Int}(
        ContinuousRandomVariable(draw, schedule);
        seed = BranchingSeed(1, "draw"),
        title = "Scheduled Draw",
        dimensions = ["draw" => ""],
    )

    # The schedule appears only on the random variable. It should still create exact hard
    # boundaries, while the final value is held through a partial interval at the end of
    # the requested simulation.
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_random_variables = (;
                draw = described_draw,
            ),
        ),
        options = SimOptions(;
            solver = SystemsOfSystems.Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test first.(draw_inputs) == [0//1, 3//10, 3//5, 9//10]
    @test last.(draw_inputs) ≈ fill(0.3, 4)
    @test history.model.draw == 4

end

@testset "continuous random variables require schedules" begin

    description = ModelDescription(;
        continuous_random_variables = (;
            draw = (rng, t_km1, dt_f) -> 0.,
        ),
    )
    err = try
        initialize(description)
    catch err
        err
    end

    @test err isa ArgumentError
    @test occursin("/continuous_random_variables/draw", sprint(showerror, err))

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
                t = (0//1, t_final),
                seed,
                init_fcn = (args...) -> ModelDescription(;
                    type = ContinuousRandomWalk,
                    continuous_states = (;
                        x = 0.,
                    ),
                    continuous_random_variables = (;
                        nu = ContinuousWhiteNoise(
                            sigma_noise,
                            RegularSchedule(1//10),
                        ),
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
