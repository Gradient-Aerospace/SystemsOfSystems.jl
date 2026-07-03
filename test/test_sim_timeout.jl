module TestSimTimeout

using SystemsOfSystems
using Test

@testset "SimTimeout" begin

    history, t_final, model_final = simulate(
        nothing;
        t = 0 : 1 : 10,
        init_fcn = (args...) -> ModelDescription(;
            discrete_states = (;
                t = 0.,
            ),
        ),
        updates_fcn = (t, model) -> begin
            sleep(2) # Sleep for 2s, so the whole sim should take at least 20s.
            UpdatesOutput(;
                updates = (;
                    t = float(t),
                ),
            )
        end,
        options = SimOptions(;
            hooks = [
                Hooks.SimTimeoutOptions(10.),
            ],
        ),
    )

    # Make sure it was the hook that stopped the sim.
    @test history.stop isa SystemsOfSystems.HookRequestedStop

    # It should have ended before the last time we asked for.
    @test t_final < 10

    # Make sure the stop reason and simulate call fundamentally report the same end time.
    @test history.stop.t == t_final

    # On the last step, the discrete update should not have run.
    @test model_final.t < float(history.stop.t)

end

end
