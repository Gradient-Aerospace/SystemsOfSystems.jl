using Test
using SystemsOfSystems

@testset "test exponential" begin

    # We'll simulate a pure exponential decay and compare to the known answer.
    time_constant = 2.
    t_end = 5.
    history, t, x = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            constants = (;
                time_constant,
            ),
            continuous_states = (;
                x = 1.,
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(
                rates = (;
                    x = -1/model.time_constant * model.x,
                ),
            )
        end,
        updates_fcn = (t, model) -> UpdatesOutput(),
        t = (0, t_end),
    )
    @test history["/"]["x"].data[1] == 1.
    @test history["/"]["x"].data[end] ≈ exp(-t_end/time_constant) atol=1e-4

end

@testset "closed loop control" begin

    # We'll simulate a closed-loop control system to test hybrid systems.
    dt = 0.05
    t_end = 5.
    history, t, x = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(
            constants = (;
                dt = dt,
                kp = 8.,
                kd = 4.,
                mass = 1.,
            ),
            continuous_states = (;
                position = 1.,
                velocity = 0.,
            ),
            continuous_outputs = (;
                acceleration = 0.,
            ),
            discrete_states = (;
                force = 0.,
            ),
            discrete_outputs = (;
                control_error = 0.,
            ),
            t_next = 0.05,
        ),
        rates_fcn = (t, model) -> begin
            acceleration = model.force / model.mass
            RatesOutput(
                rates = (;
                    position = model.velocity,
                    velocity = acceleration,
                ),
                outputs = (;
                    acceleration,
                )
            )
        end,
        updates_fcn = (t, model) -> begin
            if is_regular_step_triggering(t, model.dt)
                UpdatesOutput(
                    updates = (;
                        force = -model.kp * model.position - model.kd * model.velocity,
                    ),
                    outputs = (;
                        control_error = model.position,
                    ),
                    t_next = t + model.dt,
                )
            else
                UpdatesOutput()
            end
        end,
        t = (0, t_end),
    )

    # It started in the right place.
    @test history["/"]["position"].data[1] == 1.

    # THe control system more or less worked.
    @test abs(history["/"]["position"].data[end]) < 0.1

    # We got the expected number of discrete steps.
    @test length(history["/"]["force"].data) == t_end / dt + 1
    @test length(history["/"]["control_error"].data) == t_end / dt + 1

end

# TODO: Test systems of systems, loggers, monitors, random variables, VariableDescriptions, multiple types, very long sims with lots of steps...
