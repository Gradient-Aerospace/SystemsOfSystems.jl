module TestRandomVariableLifecycle

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

struct ControlledErrorState
    value::Float64
end

Base.:+(a::ControlledErrorState, b::ControlledErrorState) =
    ControlledErrorState(a.value + b.value)
Base.:*(scale::Number, state::ControlledErrorState) =
    ControlledErrorState(scale * state.value)

const error_evaluations = Ref(0)

function SystemsOfSystems.normalized_variable_error(
    value::ControlledErrorState,
    embedded_value::ControlledErrorState,
    absolute_tolerance,
    relative_tolerance,
)
    error_evaluations[] += 1
    return error_evaluations[] == 1 ? 2. : 0.
end

@testset "random draws follow attempted and accepted steps" begin

    # The first adaptive attempt is rejected by ControlledErrorState's error method. Every
    # attempt should redraw continuous variables, while discrete variables should be drawn
    # only after an endpoint has been accepted.
    error_evaluations[] = 0
    continuous_draws = Tuple{Any, Float64}[]
    discrete_draw_times = Any[]
    update_times = Any[]
    continuous_draw = (rng, t_km1, dt_f) -> begin
        push!(continuous_draws, (t_km1, dt_f))
        return length(continuous_draws)
    end
    discrete_draw = (rng, t) -> begin
        push!(discrete_draw_times, t)
        return length(discrete_draw_times)
    end

    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = ControlledErrorState(0.),),
            continuous_random_variables = (; continuous_draw,),
            discrete_random_variables = (; discrete_draw,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = ControlledErrorState(1.),),
        ),
        updates_fcn = (t, model) -> begin
            push!(update_times, t)
            return nothing
        end,
        options = SimOptions(;
            log = nothing,
            solver = Solvers.DormandPrince54Options(;
                initial_dt = 1,
                max_dt = 1,
                abs_tol = 1.,
                rel_tol = 0.,
            ),
        ),
    )

    @test succeeded(history)
    @test error_evaluations[] == length(update_times) + 1

    # Both random-variable kinds receive one initialization draw. Thereafter, continuous
    # draws correspond to attempts and discrete draws correspond to accepted updates.
    @test length(continuous_draws) == length(update_times) + 2
    @test continuous_draws[2][1] == continuous_draws[3][1] == 0
    @test continuous_draws[3][2] < continuous_draws[2][2]
    @test first(discrete_draw_times) == 0
    @test discrete_draw_times[2:end] == update_times

end

end # module TestRandomVariableLifecycle
