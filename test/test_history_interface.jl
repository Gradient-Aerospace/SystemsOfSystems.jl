module TestHistoryInterface

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

@testset "SimHistory log interface" begin

    # A root state and output exercise the two relevant history categories. The child state
    # exercises recursive gathering, while the constant confirms that only time series are
    # returned.
    history = simulate(
        nothing;
        t = (0, 1),
        init_fcn = (args...) -> ModelDescription(;
            constants = (; constant = 3.,),
            continuous_states = (; x = 0.,),
            continuous_outputs = (; double_x = 0.,),
            models = (;
                child = ModelDescription(;
                    continuous_states = (; y = 1.,),
                ),
            ),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            rates = (; x = 1.,),
            outputs = (; double_x = 2 * model.x,),
            models = (;
                child = RatesOutput(; rates = (; y = -1.,)),
            ),
        ),
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test history["/child"] === history.log["/child"]
    @test collect(keys(history)) == collect(keys(history.log))
    @test collect(values(history)) == collect(values(history.log))
    @test collect(pairs(history)) == collect(pairs(history.log))

    time_series = Logs.gather_all_time_series(history)
    @test collect(keys(time_series)) == [":x", ":double_x", "/child:y"]
    @test time_series[":x"] === history["/"]["x"]
    @test time_series[":double_x"] === history["/"]["double_x"]
    @test time_series["/child:y"] === history["/child"]["y"]
    @test !haskey(time_series, ":constant")

end

end # module TestHistoryInterface
