module TestHooks

using SystemsOfSystems
using Test

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

@testset "Hooks use the (t, model) inputs" begin

    # We'll mutate these with the hook.
    t_storage       = Float64[]
    x_storage       = Float64[]
    t_final_storage = Float64[]
    x_final_storage = Float64[]

    # Create a simulation that produces a sinusoid and has the hook.
    history, t_final, model_final = simulate(
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
    @test only(t_final_storage) == t_final
    @test only(x_final_storage) == model_final.x

end

end
