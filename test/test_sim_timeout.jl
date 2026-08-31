module TestSimTimeout

using Test
using SystemsOfSystems

@testset "SimTimeout" begin

    history = simulate(
        nothing;
        t = 0 : 0.1 : 1,
        init_fcn = (args...) -> ModelDescription(;
            discrete_states = (;
                t = 0.,
            ),
        ),
        updates_fcn = (t, model) -> begin
            sleep(0.2) # Sleep for 0.2s, so the whole sim should take at least 2s.
            UpdatesOutput(;
                updates = (;
                    t = float(t),
                ),
            )
        end,
        options = SimOptions(;
            hooks = [
                Hooks.SimTimeoutOptions(1.),
            ],
        ),
    )

    # Make sure it was the hook that stopped the sim.
    @test history.stop isa SystemsOfSystems.HookRequestedStop

    # It should have ended before the last time we asked for.
    @test history.t_stop < 1

    # Make sure the stop reason and simulate call fundamentally report the same end time.
    @test history.stop.t == history.t_stop

    # On the last step, the discrete update should have run.
    @test history.model.t == float(history.stop.t)

end

end
