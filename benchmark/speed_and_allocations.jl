# Run this like so:
#
#  julia --project=. benchmark/speed_and_allocations.jl
#
module BenchmarkSpeedAndAllocations

using Random: Xoshiro, randn
import Dimensions
import SystemsOfSystems
using SystemsOfSystems: ModelDescription, RatesOutput, UpdatesOutput, RegularSchedule,
    Solvers, Logs, SimOptions, LoggingPolicies, Samplers

include("../test/control_system_demo.jl")
using .ControlSystemDemo

const out_dir = joinpath(@__DIR__, "out")
mkpath(out_dir)

#########################
# ManyClosedLoopSystems #
#########################

# The only purpose of this is to make the model tree deeper. We'll just hold on to a bunch
# of ClosedLoopSystems.
@kwdef struct ManyClosedLoopSystemsSpecs
    a::ClosedLoopSystemSpecs
    b::ClosedLoopSystemSpecs
    c::ClosedLoopSystemSpecs
    d::ClosedLoopSystemSpecs
    e::ClosedLoopSystemSpecs
end

@kwdef struct ManyClosedLoopSystems
    a::ClosedLoopSystem
    b::ClosedLoopSystem
    c::ClosedLoopSystem
    d::ClosedLoopSystem
    e::ClosedLoopSystem
end

function ControlSystemDemo.init(t, specs::ManyClosedLoopSystemsSpecs, seed)
    return ModelDescription(;
        type = ManyClosedLoopSystems,
        models = (;
            a = ControlSystemDemo.init(t, specs.a, seed / "a"),
            b = ControlSystemDemo.init(t, specs.b, seed / "b"),
            c = ControlSystemDemo.init(t, specs.c, seed / "c"),
            d = ControlSystemDemo.init(t, specs.d, seed / "d"),
            e = ControlSystemDemo.init(t, specs.e, seed / "e"),
        ),
    )
end

function ControlSystemDemo.rates(t, system::ManyClosedLoopSystems)
    return RatesOutput(;
        models = (;
            a = ControlSystemDemo.rates(t, system.a),
            b = ControlSystemDemo.rates(t, system.b),
            c = ControlSystemDemo.rates(t, system.c),
            d = ControlSystemDemo.rates(t, system.d),
            e = ControlSystemDemo.rates(t, system.e),
        ),
    )
end

function ControlSystemDemo.updates(t, system::ManyClosedLoopSystems)
    return UpdatesOutput(;
        models = (;
            a = ControlSystemDemo.updates(t, system.a),
            b = ControlSystemDemo.updates(t, system.b),
            c = ControlSystemDemo.updates(t, system.c),
            d = ControlSystemDemo.updates(t, system.d),
            e = ControlSystemDemo.updates(t, system.e),
        ),
    )
end

#########################
# Speed and Allocations #
#########################

function make_inputs(system_specs, solver, log, t_end)
    return (;
        user_data = system_specs,
        t = (0, t_end),
        init_fcn = ControlSystemDemo.init,
        rates_fcn = ControlSystemDemo.rates,
        updates_fcn = ControlSystemDemo.updates,
        close_fcn = (t, model) -> nothing,
        seed = 0,
        options = SimOptions(;
            solver,
            log,
            time_dimension = "Time" => "s",
        ),
    )
end

function run_for_warmup(system_specs, solver, log)
    inputs = make_inputs(system_specs, solver, log, 1)
    runtime = SystemsOfSystems.make_runtime(inputs)
    GC.gc()
    loop_outputs = SystemsOfSystems.loop!(runtime)
    result = SystemsOfSystems.tear_down(runtime, loop_outputs)
    @assert result.history.stop isa SystemsOfSystems.ReachedEndTime
    return (result.history, result.t_final, result.final_model)
end

function run_for_timing(system_specs, solver, log, t_end)
    inputs = make_inputs(system_specs, solver, log, t_end)
    runtime = SystemsOfSystems.make_runtime(inputs)
    GC.gc()
    @time loop_outputs = SystemsOfSystems.loop!(runtime)
    result = SystemsOfSystems.tear_down(runtime, loop_outputs)
    @assert result.history.stop isa SystemsOfSystems.ReachedEndTime
    return (result.history, result.t_final, result.final_model)
end

function warm_up_then_time(system_specs, solver, log, t_end)
    run_for_warmup(system_specs, solver, log)
    run_for_timing(system_specs, solver, log, t_end)
end

# We wrap this in a function so we aren't dumping a bunch of things into this module.
function time_simulations()

    for solver_type in ["rk4", "dp54"]

        for log_type in ["ram", "null"] # Deliberately excludes hdf5 for now.

            dt_rk4 = 0.02
            solver = if solver_type == "dp54"
                Solvers.DormandPrince54Options()
            elseif solver_type == "rk4"
                Solvers.RungeKutta4Options(; dt = dt_rk4)
            else
                error("Unknown solver type: $solver_type")
            end

            # We'll use a logging policy to make sure sampling is tested as part of the
            # benchmark.
            logging_policy = LoggingPolicies.RegexLoggingPolicy(;
                rules = [
                    # We'll log /a (and everything below it) at a reduced rate.
                    r"^/a$" => LoggingPolicies.ModelLoggingPolicy(;
                        sampler = Samplers.RegularSampler(;
                            period = 1//1,
                        ),
                    ),
                ],
                # Everything else can log completely.
                default = LoggingPolicies.AllPassModelLoggingPolicy(),
            )

            log = if log_type == "ram"
                Logs.BasicLogOptions(; logging_policy)
            elseif log_type == "hdf5"
                Logs.HDF5LogOptions(;
                    filename = joinpath(out_dir, "speed_and_allocation_logs.h5"),
                    logging_policy,
                )
            elseif log_type == "null"
                Logs.NullLogOptions()
            else
                error("Unknown log type: $log_type")
            end

            rng = Xoshiro(1)

            # Set the parameters for all of the models.
            function make_closed_loop_system(position, velocity)
                return ClosedLoopSystemSpecs(
                    plant = PlantSpecs(
                        mass = 1.,
                        initial_position = position,
                        initial_velocity = velocity,
                        acceleration_noise_sigma = 0.1,
                    ),
                    sensor = SensorSpecs(
                        schedule = RegularSchedule(0.1),
                        sigma_noise = 0.,
                        sigma_bias = 0.,
                    ),
                    target = ConstantTargetSpecs(
                        constant_position = 1.,
                    ),
                    controller = PDControllerSpecs(
                        schedule = RegularSchedule(0.1),
                        p = 8.,
                        d = 4.,
                        initial_position = 0.,
                        initial_command = 0.,
                    ),
                    actuator = ActuatorSpecs(
                        time_constant = 0.2,
                        initial_command = 0.,
                        initial_response = 0.,
                    ),
                )
            end

            system_specs = ManyClosedLoopSystemsSpecs(;
                a = make_closed_loop_system(randn(rng), randn(rng)),
                b = make_closed_loop_system(randn(rng), randn(rng)),
                c = make_closed_loop_system(randn(rng), randn(rng)),
                d = make_closed_loop_system(randn(rng), randn(rng)),
                e = make_closed_loop_system(randn(rng), randn(rng)),
            )

            println("solver = $solver_type, log = $log_type")
            simout = warm_up_then_time(system_specs, solver, log, 100)

        end

    end

end

end # BenchmarkSpeedAndAllocations

# Now actually run the timing.
BenchmarkSpeedAndAllocations.time_simulations()
