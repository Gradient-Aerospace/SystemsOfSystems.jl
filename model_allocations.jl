using Random
using SystemsOfSystems: SystemsOfSystems, ModelStateDescription, model, copy_model_state_description_except

@kwdef struct MyType
    a::Int64
    b::Float64
    c::Float64
    x::Float64
end

# # This is cheating, but that's probably fine. (Do we need to .* in the solver?)
# Base.:*(a::Real, b::NTuple{N, T}) where {N, T <: Real} = a .* b
# Base.:+(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T <: Real} = a .+ b

@kwdef struct MyOuterType
    c::Float64
    inner::MyType
end

function set_up_msd(k)
    return ModelStateDescription{MyOuterType}(;
        constants = (;
            c = 6.,
        ),
        models = (;
            inner = ModelStateDescription{MyType}(;
                constants = (;
                    a = k,
                ),
                continuous_states = (;
                    x = 1.,
                ),
                discrete_states = (;
                    b = float(k),
                ),
                discrete_random_variables = (;
                    c = 0.,
                ),
                t_next = 0//1,
            ),
        ),
    )
end

function populate_msds!(msds)
    for k in eachindex(msds)
        msds[k] = set_up_msd(k)
    end
end

function populate_models!(models, msds)
    for k in eachindex(models)
        models[k] = model(msds[k])
    end
end

function copy_msds!(copied_msds, msds)
    for k in eachindex(msds)
        copied_msds[k] = copy_model_state_description_except(
            msds[k];
            constants = (;
               c = 3.14,
            ),
            t_next = 0//1,
        )
    end
end

function populate_t_next!(t_next, msd)
    for k in eachindex(t_next)
        t_next[k] = SystemsOfSystems.find_soonest_t_next(1//1, msd)
    end
end

prototype = set_up_msd(1)

n = 100
msds = Vector{typeof(prototype)}(undef, n)
models = Vector{MyOuterType}(undef, n)

println("ModelStateDescription")
populate_msds!(msds)
@time populate_msds!(msds)

println("model")
populate_models!(models, msds)
@time populate_models!(models, msds)

copied_msds = Vector{typeof(prototype)}(undef, n)

println("copy_model_state_description_except")
copy_msds!(copied_msds, msds)
@time copy_msds!(copied_msds, msds)

println("is_regular_step_triggering")
SystemsOfSystems.is_regular_step_triggering(1//1, 1//1, 0//1)
@time SystemsOfSystems.is_regular_step_triggering(1//1, 1//1, 0//1)

msd = msds[1]
@show isbits(msd)

println("recursive_reduce")
t_next = Vector{Rational{Int64}}(undef, 1)
populate_t_next!(t_next, msd)
@time populate_t_next!(t_next, msd)

function my_init()
    rng = Xoshiro(1)
    return SystemsOfSystems.ModelDescription(;
        type = MyOuterType,
        constants = (;
            c = 6.,
        ),
        models = (;
            inner = SystemsOfSystems.ModelDescription(;
                type = MyType,
                constants = (;
                    a = 0,
                ),
                continuous_states = (;
                    x = 1.,
                ),
                discrete_states = (;
                    b = 0.,
                ),
                discrete_random_variables = (;
                    c = (t) -> randn(rng),
                ),
                t_next = 0//1,
            ),
        ),
    )
end
ommd = my_init()

msd = prototype

function draw_wd_here(ommd, msd)
    SystemsOfSystems.draw_wd(0//1, ommd, msd)
    return nothing
end

function draw_wc_here(ommd, msd)
    SystemsOfSystems.draw_wc(0//1, 1//1, ommd, msd)
    return nothing
end

println("draw_wd")
draw_wd_here(ommd, msd)
@time draw_wd_here(ommd, msd)

println("draw_wc")
draw_wc_here(ommd, msd)
@time draw_wc_here(ommd, msd)

function update_msd(msd, k)
    uo = SystemsOfSystems.UpdatesOutput(;
        models = (;
            discrete_states = (;
                b = float(k),
            ),
        ),
    )
    return SystemsOfSystems.update(msd, uo)
end

function update_msds!(msds_out, msds_in)
    for k in eachindex(msds_out)
        msds_out[k] = update_msd(msds_in[k], k)
    end
end

println("update")
updated_msds = Vector{typeof(prototype)}(undef, n)
update_msds!(updated_msds, msds)
@time update_msds!(updated_msds, msds)

function rates(t, model)
    return SystemsOfSystems.RatesOutput(;
        models = (;
            inner = SystemsOfSystems.RatesOutput(;
                rates = (;
                    x = 2.,
                ),
            ),
        ),
    )
end

function propagate_msd(msd, k)
    rates_output = rates(0//1, model(msd))
    return SystemsOfSystems.Solvers.propagate(msd, 1//1, rates_output)
end

function propagate_msds!(msds_out, msds_in)
    for k in eachindex(msds_out)
        msds_out[k] = propagate_msd(msds_in[k], k)
    end
end

println("propagate")
updated_msds = Vector{typeof(prototype)}(undef, n)
propagate_msds!(updated_msds, msds)
@time propagate_msds!(updated_msds, msds)

function propagate2_msd(msd, k)
    rates_output = rates(0//1, model(msd))
    return SystemsOfSystems.Solvers.propagate(msd, (1., 1.), (rates_output, rates_output))
end

function propagate2_msds!(msds_out, msds_in)
    for k in eachindex(msds_out)
        msds_out[k] = propagate2_msd(msds_in[k], k)
    end
end

println("propagate2")
updated_msds = Vector{typeof(prototype)}(undef, n)
propagate2_msds!(updated_msds, msds)
@time propagate2_msds!(updated_msds, msds)

solver = SystemsOfSystems.Solvers.create_solver(SystemsOfSystems.Solvers.RungeKutta4Options(; dt = 1//1), msd)

function foo(ommd, solver, msd, rates)
    solution = SystemsOfSystems.Solvers.solve(ommd, solver, 0//1, 1//1, msd, rates, 100//1)
    return isbits(solution)
end

println("solve")
bar = foo(ommd, solver, msd, rates)
@time bar = foo(ommd, solver, msd, rates)
@show bar
