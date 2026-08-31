module TestOutputValidation

using Test
using SystemsOfSystems
using SystemsOfSystems: Solvers

function capture_exception(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

function test_description(; models = (;))
    return ModelDescription(;
        continuous_states = (; continuous_state = 0.),
        discrete_states = (; discrete_state = 0,),
        continuous_outputs = (; continuous_output = 0.),
        discrete_outputs = (; discrete_output = 0,),
        models,
    )
end

@testset "user-function outputs reject undeclared names" begin

    # Rates and updates may omit any declared field, but names that are present must belong
    # to the corresponding section of the model description. A nested rates typo also
    # verifies that validation follows the returned model hierarchy and reports its path.
    description = test_description(; models = (; child = test_description(),))
    artifacts = SystemsOfSystems.create_initialization_artifacts(
        description,
        SystemsOfSystems.initialization_context(),
    )
    ommd = artifacts.ommd
    @test ommd.model_path == "/"
    @test ommd.models.child.model_path == "/child"

    rates_output = RatesOutput(;
        models = (; child = RatesOutput(; outputs = (; typo = 1.,)),),
    )
    rates_exception = capture_exception() do
        SystemsOfSystems.validate_rates_output(ommd, rates_output)
    end
    @test rates_exception isa ArgumentError
    @test occursin("Model /child", sprint(showerror, rates_exception))
    @test occursin("unexpected field `typo`", sprint(showerror, rates_exception))
    @test occursin("`RatesOutput.outputs`", sprint(showerror, rates_exception))

    updates_output = UpdatesOutput(; updates = (; typo = 1.,))
    updates_exception = capture_exception() do
        SystemsOfSystems.validate_updates_output(ommd, updates_output)
    end
    @test updates_exception isa ArgumentError
    @test occursin("unexpected field `typo`", sprint(showerror, updates_exception))
    @test occursin("`UpdatesOutput.updates`", sprint(showerror, updates_exception))

    @test isnothing(SystemsOfSystems.validate_rates_output(ommd, RatesOutput()))

    SystemsOfSystems.close_resources(artifacts.manager)

end

@testset "a malformed output retains the last committed history" begin

    # The first evaluation at t = 1 completes the step from zero. The next evaluation at
    # that same time begins a new step and returns a misspelled output name. Validation
    # should stop there without discarding the state and history already committed at t = 1.
    n_rates_at_one = Ref(0)
    history = @test_logs (:error,) simulate(
        nothing;
        t = (0, 2),
        init_fcn = (args...) -> ModelDescription(;
            continuous_states = (; x = 0.),
            continuous_outputs = (; observed_x = 0.),
        ),
        rates_fcn = (t, model) -> begin

            if t == 1.
                n_rates_at_one[] += 1
                if n_rates_at_one[] == 2
                    return RatesOutput(;
                        rates = (; x = 1.,),
                        outputs = (; observd_x = model.x,),
                    )
                end
            end

            return RatesOutput(;
                rates = (; x = 1.,),
                outputs = (; observed_x = model.x,),
            )

        end,
        options = SimOptions(;
            solver = Solvers.RungeKutta4Options(; dt = 1),
        ),
    )

    @test history.t_stop == 1
    @test history.model.x ≈ 1.
    @test history.stop isa SystemsOfSystems.EncounteredError
    @test history["/"]["x"].time[end] == 1.
    @test history["/"]["x"].data[end] == history.model.x
    @test occursin(
        "unexpected field `observd_x` in `RatesOutput.outputs`",
        sprint(showerror, history.stop.exception),
    )

end

end # module TestOutputValidation
