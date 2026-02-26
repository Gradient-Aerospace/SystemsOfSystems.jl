module MyModule

export foo, copy_it_1!, copy_it_2!, copy_it_3!, copy_it_4!, copy_it_5!
export MyStruct, SubModel

@kwdef mutable struct SubModel
    value::Float64
end

@kwdef mutable struct MyStruct{NT}
    models::NT
    # x::Float64
    # y::Float64
end

# Doesn't allocate:
# function Base.getproperty(m::MyStruct, f::Symbol)
#     if hasfield(MyStruct, f)
#         return getfield(m, f)
#     end
#     return getproperty(getfield(m, :models), f)
# end

# Allocates:
# function Base.getproperty(m::MyStruct, f::Symbol)
#     if hasfield(MyStruct, f)
#         return getfield(m, f)
#     elseif hasfield(fieldtype(typeof(m), :models), f)
#         return getproperty(getfield(m, :models), f)
#     end
#     # error("Nope.")
#     # return getproperty(getfield(m, :models), f)
# end

# Allocates:
# function Base.getproperty(m::MyStruct{NT}, f::Symbol) where {NT}
#     if hasfield(MyStruct, f)
#         return getfield(m, f)
#     elseif hasfield(NT, f)
#         return getproperty(getfield(m, :models), f)
#     end
# end

# Doesn't allocate:
# function Base.getproperty(m::MyStruct{NT}, f::Symbol) where {NT}
#     if hasfield(NT, f)
#         return getproperty(getfield(m, :models), f)
#     end
#     return getfield(m, f)
# end

# Doesn't allocate:
# function Base.getproperty(m::MyStruct{NT}, f::Symbol) where {NT}
#     if hasfield(NT, f)
#         return getproperty(getfield(m, :models), f)
#     end
#     try
#         return getfield(m, f)
#     catch err
#         rethrow("I don't have field $f.")
#     end
# end

# Allocates:
# function Base.getproperty(m::MyStruct{NT}, f::Symbol) where {NT}
#     if hasfield(MyStruct, f)
#         return getfield(m, f)
#     end
#     @assert hasfield(NT, f) "Nope."
#     return getproperty(getfield(m, :models), f)
#     # try

#     # catch err
#     #     rethrow("I don't have field $f.")
#     # end
# end

# Doesn't allocate:
function Base.getproperty(m::MyStruct{NT}, f::Symbol) where {NT}
    if hasfield(typeof(m), f)
        return getfield(m, f)
    end
    try
        models = getfield(m, :models)
        return models[f]
    catch
        error("I don't have field $f. I have: $(fieldnames(NT)).")
    end
end

function copy_it_1!(m::MyStruct, n)
    for (k, v) in pairs(n.stuff)
        # setfield!(m, k, v)
        # m.models[k].value = v
        getproperty(m, k).value = v
    end
end

# function copy_it_2!(m::MyStruct, n::T) where {T}
#     for f in fieldnames(T)
#         setfield!(m, f, n.stuff[f])
#     end
# end

function copy_it_3!(m::MyStruct, n)
    models = m.models
    models.x.value = n.stuff.x
    models.y.value = n.stuff.y
end

# function copy_it_4!(m::MyStruct, n::T) where {T}
#     foreach(fieldnames(T)) do f
#         setfield!(m, f, n.stuff[f])
#     end
# end

function copy_it_5!(m::MyStruct, n)
    foreach(fieldnames(typeof(n.stuff))) do f
        # setfield!(m, f, n.stuff[f])
        # getfield(m.models, f).value = n.stuff[f]
        getproperty(m, f).value = n.stuff[f]
    end
end

function foo(f!, m)
    for x in 1. : 100000.
        n = (; stuff = (; x, y = 2x), )
        f!(m, n)
    end
    return m
end

end

using .MyModule
m = MyStruct(; models = (; x = SubModel(0.), y = SubModel(0.)), )
foo(copy_it_1!, m)
# foo(copy_it_2!)
foo(copy_it_3!, m)
# foo(copy_it_4!)
foo(copy_it_5!, m)
@time foo(copy_it_1!, m)
# @time foo(copy_it_2!)
@time foo(copy_it_3!, m)
# @time foo(copy_it_4!)
@time foo(copy_it_5!, m)
