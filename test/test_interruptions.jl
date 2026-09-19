module TestInterruptions

using Test
using SystemsOfSystems
using SystemsOfSystems: Interrupted, EncounteredError, SimulationTimes

struct ClosingHookOptions <: Hooks.AbstractHookOptions
    name::Symbol
    closed::Vector{Symbol}
    endpoints::Vector{Tuple{SimulationTimes.ExactTime, Float64}}
end

struct ClosingHook <: Hooks.AbstractHook
    options::ClosingHookOptions
end

Hooks.create_hook(options::ClosingHookOptions, t, model) = ClosingHook(options)
Hooks.update_hook!(::ClosingHook, t, model) = Hooks.HookOutputs()

function Hooks.close_hook!(hook::ClosingHook, t, model)
    push!(hook.options.closed, hook.options.name)
    push!(hook.options.endpoints, (t, model.x))
    return nothing
end

function interrupted_run(; log = Logs.BasicLogOptions(), exception = InterruptException())

    # The first RK4 step is accepted at 0.5. The next midpoint is provisional and throws.
    # Closure records let us verify teardown without depending on operating-system signals.
    closed = Symbol[]
    endpoints = Tuple{SimulationTimes.ExactTime, Float64}[]
    evaluations = Float64[]
    history = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; observed_x = 0.),
            resources = (;
                first = Resource(;
                    open_args = (),
                    open_fcn = inputs -> :first_resource,
                    close_fcn = name -> push!(closed, name),
                ),
                second = Resource(;
                    open_args = (),
                    open_fcn = inputs -> :second_resource,
                    close_fcn = name -> push!(closed, name),
                ),
            ),
        ),
        t = (0, 2),
        rates_fcn = (t, model) -> begin

            push!(evaluations, t)
            t > 0.5 && throw(exception)
            return RatesOutput(;
                rates = (; x = 1.),
                outputs = (; observed_x = model.x),
            )

        end,
        options = SimOptions(;
            log,
            solver = Solvers.RungeKutta4Options(; dt = 1//2),
            hooks = [
                ClosingHookOptions(:first_hook, closed, endpoints),
                ClosingHookOptions(:second_hook, closed, endpoints),
            ],
        ),
    )
    return (; history, closed, endpoints, evaluations)

end

function sampled_log(sampler)
    return Logs.BasicLogOptions(;
        logging_policy = LoggingPolicies.UniformLoggingPolicy(
            LoggingPolicies.ModelLoggingPolicy(; sampler),
        ),
    )
end

@testset "interrupted RK stage preserves the accepted endpoint" begin

    for sampler in (Samplers.CompleteSampler(), Samplers.RegularSampler(1))

        result = @test_logs interrupted_run(; log = sampled_log(sampler))
        history = result.history
        @test history.stop isa Interrupted
        @test history.stop.t === history.t_stop === 1//2
        @test history.model.x ≈ 0.5
        @test succeeded(history)
        @test occursin("interrupted", SystemsOfSystems.describe(history.stop))

        # Interruption preserves the normal sampling policy, including off-grid omissions.
        expected = sampler isa Samplers.CompleteSampler ? [0., 0.5] : [0.]
        @test history["/"]["x"].time == expected
        @test history["/"]["x"].data ≈ expected
        @test history["/"]["observed_x"].time == [0.]
        @test result.evaluations == [0., 0.25, 0.25, 0.5, 0.5, 0.75]

        # Teardown sees only the committed model and closes every hook and resource.
        @test result.closed == [
            :second_hook, :first_hook, :second_resource, :first_resource,
        ]
        @test result.endpoints == fill((history.t_stop, history.model.x), 2)

    end

end

@testset "interruption honors disabled state logging" begin

    for sampler in (
        Samplers.NullSampler(),
        Samplers.SamplingDirective(; log_states = false, log_outputs = true),
    )

        history = interrupted_run(; log = sampled_log(sampler)).history
        @test history.stop isa Interrupted
        @test isempty(history["/"]["x"].time)

    end

    # Classification and the accepted endpoint are also available without logging.
    for log in (nothing, Logs.NullLogOptions())
        history = interrupted_run(; log).history
        @test history.stop isa Interrupted
        @test history.t_stop == 1//2
    end

end

@testset "unexpected RK exceptions remain failures" begin

    # Changing the exception changes its classification, not the accepted state or logs.
    for sampler in (Samplers.CompleteSampler(), Samplers.RegularSampler(1))

        log = sampled_log(sampler)
        exception = ErrorException("A model failed.")
        result = @test_logs (:error,) interrupted_run(; log, exception)
        interrupted = @test_logs interrupted_run(; log)
        @test result.history.stop isa EncounteredError
        @test result.history.stop.exception === exception
        @test !succeeded(result.history)
        @test result.history.t_stop == interrupted.history.t_stop == 1//2
        @test result.history.model.x == interrupted.history.model.x
        @test result.closed == interrupted.closed
        @test result.endpoints == interrupted.endpoints
        @test result.evaluations == interrupted.evaluations
        for name in ("x", "observed_x")
            @test result.history["/"][name].time == interrupted.history["/"][name].time
            @test result.history["/"][name].data == interrupted.history["/"][name].data
        end

    end

end

@testset "terminal interruption retains post-update state" begin

    # The terminal state phase already recorded the committed update before rates throw.
    # Interruption preserves this state and does not add another sample or any output.
    history = simulate(
        nothing;
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; observed_x = 0.),
        ),
        t = (0, 1),
        rates_fcn = (t, model) -> begin

            model.x == 10. && throw(InterruptException())
            return RatesOutput(;
                rates = (; x = 1.),
                outputs = (; observed_x = model.x),
            )

        end,
        updates_fcn = (t, model) -> UpdatesOutput(; updates = (; x = 10.)),
        options = SimOptions(; solver = Solvers.RungeKutta4Options(; dt = 1)),
    )
    @test history.stop isa Interrupted
    @test history.t_stop == 1
    @test history.model.x == 10.
    @test history["/"]["x"].time == [0., 1., 1.]
    @test history["/"]["x"].data ≈ [0., 1., 10.]
    @test history["/"]["observed_x"].time == [0.]

end

@testset "interruption before the first accepted step" begin

    # With no accepted step, the initialized model and start time remain authoritative.
    history = simulate(
        nothing;
        t = (1//4, 1),
        init_fcn = (args...) -> ModelDescription(; continuous_states = (; x = 7.)),
        rates_fcn = (t, model) -> throw(InterruptException()),
        options = SimOptions(; log = sampled_log(Samplers.RegularSampler(1))),
    )
    @test history.stop isa Interrupted
    @test history.stop.t == history.t_stop == history.t_start == 1//4
    @test history.model.x == 7.
    @test isempty(history["/"]["x"].time)

end

end # TestInterruptions
