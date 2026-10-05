module TestHooks

using Test
using SystemsOfSystems

# We create a hook here to store the time and model position over time, to test that the
# hook interface is covered. Hooks don't persist after the end of the sim, so we'll use
# external storage containers for their vectors, and we'll let them mutate those storage
# containers.
struct StorageHookOptions <: Hooks.AbstractHookOptions
    t::Vector{Float64}
    x::Vector{Float64}
    t_final::Vector{Float64}
    x_final::Vector{Float64}
end
mutable struct StorageHook <: Hooks.AbstractHook
    t::Vector{Float64}
    x::Vector{Float64}
    t_final::Vector{Float64}
    x_final::Vector{Float64}
end

struct CleanupHook <: Hooks.AbstractHook
    name::Symbol
    close_order::Vector{Symbol}
    fail::Bool
end
struct CleanupHookOptions <: Hooks.AbstractHookOptions
    name::Symbol
    close_order::Vector{Symbol}
    fail::Bool
end
Hooks.create_hook(options::CleanupHookOptions, t, model) =
    CleanupHook(options.name, options.close_order, options.fail)
Hooks.update_hook!(hook::CleanupHook, t, model) = Hooks.HookOutputs()
function Hooks.create_hook(options::StorageHookOptions, t, model)
    push!(options.t, float(first(t)))
    push!(options.x, model.x)
    return StorageHook(options.t, options.x, options.t_final, options.x_final)
end
function Hooks.update_hook!(hook::StorageHook, t, model)
    push!(hook.t, float(t))
    push!(hook.x, model.x)
    return Hooks.HookOutputs()
end
function Hooks.close_hook!(hook::StorageHook, t, model)
    push!(hook.t_final, float(t))
    push!(hook.x_final, model.x)
    return nothing
end
function Hooks.close_hook!(hook::CleanupHook, t, model)
    push!(hook.close_order, hook.name)
    hook.fail && error("Expected close failure")
    return nothing
end

@testset "Hooks use the (t, model) inputs" begin

    # We'll mutate these with the hook.
    t_storage       = Float64[]
    x_storage       = Float64[]
    t_final_storage = Float64[]
    x_final_storage = Float64[]

    # Create a simulation that produces a sinusoid and has the hook.
    history = simulate(
        nothing;
        t = 0 : 0.1 : 10,
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (;
                x = 1.,
                x_dot = 0.
            ),
        ),
        rates_fcn = (t, model) -> begin
            RatesOutput(;
                rates = (;
                    x = model.x_dot,
                    x_dot = -model.x,
                ),
            )
        end,
        options = SimOptions(;
            hooks = [
                StorageHookOptions(t_storage, x_storage, t_final_storage, x_final_storage),
            ],
        ),
    )

    # The model stored everything on all steps.
    @test history["/"]["x"].time == t_storage
    @test history["/"]["x"].data == x_storage
    @test only(t_final_storage) == history.t_stop
    @test only(x_final_storage) == history.model.x

end

@testset "Hooks close after simulation errors" begin

    t_storage = Float64[]
    x_storage = Float64[]
    t_final_storage = Float64[]
    x_final_storage = Float64[]
    history = @test_logs (:error,) simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.,),
        ),
        updates_fcn = (t, model) -> error("Expected"),
        options = SimOptions(;
            hooks = [
                StorageHookOptions(t_storage, x_storage, t_final_storage, x_final_storage),
            ],
        ),
    )

    # The failure is caught by the simulation, but teardown must still close the hook with
    # the last committed time and model.
    @test !succeeded(history)
    @test only(t_final_storage) == history.t_stop
    @test only(x_final_storage) == history.model.x

end

@testset "hook cleanup is isolated and reversed" begin

    # Hooks close in reverse creation order. A failure from one hook must be logged without
    # preventing the hooks created before it from closing too.
    close_order = Symbol[]
    hooks = Hooks.AbstractHook[
        CleanupHook(:first, close_order, false),
        CleanupHook(:second, close_order, true),
        CleanupHook(:third, close_order, false),
    ]

    errors = @test_logs (:error,) SystemsOfSystems.close_hooks(hooks, 0, nothing)
    @test close_order == [:third, :second, :first]
    @test length(errors) == 1
    @test errors[1] isa SystemsOfSystems.CleanupError
    @test occursin("hook 2", SystemsOfSystems.cleanup_context(errors[1]))
    @test occursin("Expected close failure", sprint(showerror, errors[1].exception))

end

@testset "simulation returns hook cleanup failures independently of its stop" begin

    close_order = Symbol[]
    options = SimOptions(; hooks = Hooks.AbstractHookOptions[
        CleanupHookOptions(:first, close_order, false),
        CleanupHookOptions(:second, close_order, true),
        CleanupHookOptions(:third, close_order, false),
    ])
    run(updates_fcn) = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(; continuous_states = (; x = 0.)),
        rates_fcn = (t, model) -> RatesOutput(; rates = (; x = 1.)),
        updates_fcn,
        options,
    )

    completed = @test_logs (:error,) run((args...) -> nothing)
    @test completed.stop isa SystemsOfSystems.ReachedEndTime
    @test succeeded(completed)
    @test length(completed.cleanup_errors) == 1
    @test occursin("hook 2", SystemsOfSystems.cleanup_context(only(completed.cleanup_errors)))
    @test close_order == [:third, :second, :first]

    empty!(close_order)
    failed = @test_logs (:error,) (:error,) run((args...) -> error("Model failed"))
    @test failed.stop isa SystemsOfSystems.EncounteredError
    @test !succeeded(failed)
    @test length(failed.cleanup_errors) == 1
    @test close_order == [:third, :second, :first]

end

end
