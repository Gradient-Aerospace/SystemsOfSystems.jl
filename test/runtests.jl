module RunTests

# Start with core utilities and basic simulation behavior. The later tests exercise
# additional simulation features and options that build on that behavior.
include("test_step_triggering.jl")
include("test_simulation_times.jl")
include("test_schedules.jl")
include("test_model_filters.jl") # TODO: git mv this file.
include("test_time_series.jl")
include("test_basic_simulations.jl")
include("test_solver_options.jl")
include("test_solver_lifecycle.jl")
include("test_continuous_random_variables.jl")
include("test_random_variable_seeds.jl")
include("test_control_system_demo.jl")
include("test_updating_continuous_states.jl")
include("test_resources.jl")
include("test_hooks.jl")
include("test_clock_sync.jl")
include("test_sim_timeout.jl")

end # RunTests
