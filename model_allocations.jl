using Random
using SystemsOfSystems: SystemsOfSystems, ModelStateDescription, model, copy_model_state_description_except

@kwdef struct MyType
    a::Int64
    b::Float64
    c::Float64
    x::NTuple{3, Float64}
end

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
                    x = (1., 2., 3.),
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

function find_t_next(msd)
    return SystemsOfSystems.recursively_reduce(msd, 0//1) do el, t_next_so_far
        if iszero(el.t_next)
            return t_next_so_far
        else
            return min(t_next_so_far, rationalize(el.t_next))
        end
    end
end

function populate_t_next!(t_next, msd)
    for k in eachindex(t_next)
        t_next[k] = find_t_next(msd)
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

rng = Xoshiro(1)
ommd = SystemsOfSystems.ModelDescription(;
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
                x = (1., 2., 3.),
            ),
            discrete_states = (;
                b = 0.,
            ),
            discrete_random_variables = (;
                c = (t) -> 0.,
            ),
            t_next = 0//1,
        ),
    ),
)

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

# To test: solve
