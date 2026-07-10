module TestClockSync

using SystemsOfSystems
using Test

@testset "ClockSync" begin

    function run_sim(t, hooks)
        return simulate(
            nothing;
            t,
            init_fcn = (args...) -> ModelDescription(;
                discrete_states = (;
                    t = 0.,
                    initial_time_ns = time_ns(),
                ),
                discrete_outputs = (;
                    current_time_s = 0.,
                ),
            ),
            updates_fcn = (t, model) -> begin
                UpdatesOutput(;
                    updates = (;
                        t = float(t),
                    ),
                    outputs = (;
                        current_time_s = (time_ns() - model.initial_time_ns) * 1e-9,
                    ),
                )
            end,
            options = SimOptions(;
                hooks,
            ),
        )
    end


    # Run once to burn off compile times.
    t = 0 : 0.1 : 0.1
    history, t_final, model_final = run_sim(t, [])

    # Now run for timing's sake.
    t = 0 : 0.1 : 1
    history, t_final, model_final = run_sim(t, [Hooks.ClockSyncOptions()])

    # Make sure it went to completion.
    @test t_final == last(t)

    # Make sure samples never triggered _before_ the appropriate time.
    diffs = history["/"]["current_time_s"].data .- history["/"]["current_time_s"].time
    @test all(diffs .>= 0.)

    # Make sure things never took too long. Note: the accuracy should be way better than
    # this, but this runs in CI, and we don't want spurious failures there.
    @test all(diffs .< 0.2)

end

end
