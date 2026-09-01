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

@testset "random draws follow scheduled and accepted steps" begin

    # The first adaptive attempt is rejected by ControlledErrorState's error method.
    # Continuous values should remain fixed across the retry and every accepted numerical
    # step until their next scheduled boundary. Discrete variables are still drawn after
    # every accepted endpoint.
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
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = ControlledErrorState(0.),),
            continuous_random_variables = (;
                continuous_draw = ContinuousRandomVariable(
                    continuous_draw,
                    RegularSchedule(1),
                ),
            ),
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

    # The continuous source is drawn for [0, 1] during initialization and for [1, 2] at
    # its one nonterminal boundary. The rejected attempt does not reach the source. Both
    # random-variable kinds still receive an initialization draw.
    @test continuous_draws == [(0//1, 1.), (1//1, 1.)]
    @test first(discrete_draw_times) == 0
    @test discrete_draw_times[2:end] == update_times

end

end # module TestRandomVariableLifecycle
