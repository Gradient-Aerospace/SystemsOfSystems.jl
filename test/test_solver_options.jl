module TestSolverOptions

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

@testset "failed steps in DP54 for max_dt = $max_dt" for max_dt in (10//1, 1//10)

    # This should generate a sinusoid. When the time step is really large, it should fail
    # integration tolerances and end up with smaller steps. When it's really smaller, it
    # should observe the unnecessarily small steps.
    history, t, x = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = -model.position,
                ),
            )
        end,
        t = (0, 30),
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                initial_dt = max_dt, # Intentionally too big, to make sure this fails.
                max_dt, # Intentionally smaller than necessary.
            ),
        ),
    )

    # Make sure we're always less than the maximum.
    t = history["/"]["position"].time
    for k in 2:length(t)
        @test t[k] - t[k-1] - eps(t[k]) <= max_dt
    end

end

@testset "user-specified time steps" begin

    # This should generate a sinusoid. The max step will limit the step size here, but we'll
    # request even shorter time steps for the first several steps.
    max_dt = 1//2
    t_specified = [0.01, 0.1, 0.2, 0.3, 30.] # Includes a non-zero start
    history, t, x = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = -model.position,
                ),
            )
        end,
        t = t_specified,
        options = SimOptions(;
            solver = Solvers.DormandPrince54Options(;
                initial_dt = max_dt, # Intentionally too big, to make sure this fails.
                max_dt, # Intentionally smaller than necessary.
            ),
        ),
    )

    # Make sure we're always less than the maximum.
    t = history["/"]["position"].time
    for k in 2:length(t)
        @test t[k] - t[k-1] - eps(t[k]) <= max_dt
    end

    # Make sure the first several steps are precisely what we specified.
    @test t[1:4] == t_specified[1:4]
    @test t[end] == t_specified[end]

end

end # TestSolverOptions

