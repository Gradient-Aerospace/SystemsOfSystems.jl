module TestControlSystemDemo

using Test
using HDF5: h5open, Group
using HDF5Vectors
using SystemsOfSystems: SystemsOfSystems,
    ModelDescription, VariableDescription, BranchingSeed,
    RegularSchedule,
    Logs, SimOptions, Solvers, TimeSeries, initialize, simulate

include("control_system_demo.jl")
using .ControlSystemDemo

const out_dir = joinpath(@__DIR__, "out")
mkpath(out_dir)

################
# Test Helpers #
################

# These help us compare two HDF5 files, one created by HDF5Log and one created from
# save_log_to_hdf5.
function compare_hdf5_group(g1::Group, g2)
    @assert g2 isa Group "g1 was an Group but g2 wasn't. It was a $(typeof(g2))"
    @test keys(g1) == keys(g2)
    for k in keys(g1)
        compare_hdf5_group(g1[k], g2[k])
    end
end
function compare_hdf5_group(g1, g2)
    @test read(g1) == read(g2)
end

@testset failfast=false "control system demo with $solver_type solver, $log_type logs" for solver_type in ["rk4", "dp54"], log_type in ["ram", "hdf5"]

    dt_rk4 = 0.06 # Deliberately chosen to be inconsistent with the discrete systems' sample rates
    saved_log_path = joinpath(out_dir, "control_demo_saved_log.h5")
    solver = if solver_type == "dp54"
        Solvers.DormandPrince54Options()
    elseif solver_type == "rk4"
        Solvers.RungeKutta4Options(; dt = dt_rk4)
    end

    log = if log_type == "ram"
        Logs.BasicLogOptions()
    elseif log_type == "hdf5"
        Logs.HDF5LogOptions(joinpath(out_dir, "control_demo_logs.h5"))
    elseif log_type == "null"
        Logs.NullLogOptions()
    elseif log_type == "none"
        nothing
    end

    # Set the parameters for all of the models.
    system_specs = ControlSystemDemo.default_closed_loop_system_specs()

    # Run the sim.
    history, t, system = ControlSystemDemo.simulate_closed_loop_system(;
        system_specs, log, solver,
    )

    dt_sensor = system_specs.sensor.schedule.period
    @test t == 10
    @test history["/sensor"]["measurement"].time == collect(0. : dt_sensor : t)
    @test history["/sensor"]["measurement"].data[end].t == t

    Logs.close_log(history.log)

    if log_type == "ram"

        # Test that we can save a normal log to HDF5.
        Logs.save_log_to_hdf5(saved_log_path, history.log)
        loaded_log, = Logs.load_hdf5_log(saved_log_path)
        for model_path in keys(history.log)
            mh = history.log[model_path]
            mh2 = loaded_log[model_path]
            for var_name in keys(mh)
                var = mh[var_name]
                # @test haskey(mh2, var_name)
                var2 = mh2[var_name]
                if var isa TimeSeries # states and outputs
                    @test var2 isa TimeSeries
                    @test var.time == collect(var2.time)
                    @test var.data == collect(var2.data)
                    @test var.title == var2.title
                    @test var.dimensions == collect(var2.dimensions)
                elseif var isa VariableDescription # constants
                    @test var.value == var2 # These are undecorated. TODO: Revisit.
                elseif var isa Logs.ModelHistory # submodels
                    @test var2 isa Logs.ModelHistory
                    @test keys(var) == keys(var2)
                    # We test all of the keys of the log, so we don't need to do anything
                    # recursive here.
                else
                    @assert false "The type of the $var_name key for the $model_path entry of the log was a type we haven't accounted for."
                end
            end
        end
        Logs.close_log(loaded_log)

    elseif log_type == "hdf5"

        # Since we do the BasicLog before the HDF5Log, we can load the HDF5 log we
        # saved and compare it to the result of HDF5Log.
        f1 = h5open(joinpath(out_dir, "control_demo_logs.h5"))
        f2 = h5open(saved_log_path)
        compare_hdf5_group(f1, f2)
        close(f1)
        close(f2)

    end

    # Also, test for type stability. First, get the pieces we'll need from an internal
    # function.
    context = SystemsOfSystems.initialization_context()
    (; msd, ) = SystemsOfSystems.create_initialization_artifacts(
        ControlSystemDemo.init, system_specs, context,
    )

    # See that we can convert the model description to a model with a known type.
    @inferred SystemsOfSystems.model(msd)

    # Let's also try out `initialize`, since this might be a handy function for users to
    # debug their stuff.
    system = initialize(system_specs; init_fcn = ControlSystemDemo.init)
    @test system isa ClosedLoopSystem

    # Let's also test that we can initialize from a ModelDescription, like a user might need
    # to do during initialization.
    seed = BranchingSeed(0, "")
    model_description = ControlSystemDemo.init(0//1, system_specs, seed)
    system = initialize(model_description; seed)
    @test system isa ClosedLoopSystem

    # Our rates function should be type stable.
    @inferred ControlSystemDemo.rates(t, system)

    # Our `updates` is not type stable; on "off" samples, scheduled submodels return
    # `nothing`, which differs from their triggering UpdatesOutput type. We can nevertheless
    # verify that each off-sample call infers the narrow `Nothing` result.
    command = 0.
    meas = ControlSystemDemo.get_measurement(t, system.sensor, 0.)
    @inferred Nothing ControlSystemDemo.updates(
        t, system.sensor, meas,
    )
    @inferred Nothing ControlSystemDemo.updates(
        t, system.controller, meas, command,
    )

    # We can test this one though.
    @inferred ControlSystemDemo.updates(t, system.actuator, command)

end

end # TestControlSystemDemo
