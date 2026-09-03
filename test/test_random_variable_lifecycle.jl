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

@testset "fixed-step draws use numerical intervals" begin

    # A sparse requested-time vector should not make one continuous draw span several
    # fixed numerical steps. The first draw is the existing initialization draw.
    continuous_draws = Tuple{Any, Float64}[]
    continuous_draw = (rng, t_km1, dt_f) -> begin
        push!(continuous_draws, (t_km1, dt_f))
        return 0.
    end
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            continuous_random_variables = (; continuous_draw,),
        ),
        options = SimOptions(;
            log = nothing,
            solver = Solvers.RungeKutta4Options(; dt = 1//4),
        ),
    )

    @test succeeded(history)
    @test continuous_draws == [
        (0//1, 1.),
        (0//1, 0.25),
        (1//4, 0.25),
        (1//2, 0.25),
        (3//4, 0.25),
    ]

end

@testset "continuous draws span rejected and accepted substeps" begin

    # The first adaptive attempt is rejected by ControlledErrorState's error method. Every
    # shorter numerical attempt should retain the continuous draw committed for the
    # original interval, while discrete variables should still be drawn only after an
    # endpoint has been accepted.
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
            continuous_random_variables = (; continuous_draw,),
            discrete_random_variables = (; discrete_draw,),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = ControlledErrorState(model.continuous_draw),),
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

    # The initialization draw retains its existing unit-duration convention. The first
    # solver draw remains active through the rejected attempt and accepted substeps that
    # finish its interval. A new draw begins only after reaching t = 1.
    @test continuous_draws == [(0//1, 1.), (0//1, 1.), (1//1, 1.)]
    @test any(t -> 0 < t < 1, update_times)
    @test history.model.x == ControlledErrorState(5.)
    @test first(discrete_draw_times) == 0
    @test discrete_draw_times[2:end] == update_times

end

end # module TestRandomVariableLifecycle
