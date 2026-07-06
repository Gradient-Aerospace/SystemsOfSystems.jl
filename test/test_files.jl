module TestFiles

using SystemsOfSystems
using Test

# Start fresh by removing the files we'll output if they're already there.
const out_dir = joinpath(@__DIR__, "out")
if isdir(joinpath(out_dir, "files"))
    rm(joinpath(out_dir, "files"); recursive = true)
end

# MyModel and its functions test a simple use case for file outputs from the sim.
@kwdef struct MyModel
    x::Float64
    x_dot::Float64
    io::IOStream
end

function my_init_fcn(t, params, seed)
    return ModelDescription(;
        type = MyModel,
        continuous_states = (;
            x = 1.,
            x_dot = 0.,
        ),
        files = (;
            io = OutputFileDescription(;
                name = "my_file.txt",
            ),
        ),
    )
end

function my_rates_fcn(t, model)
    return RatesOutput(;
        rates = (;
            x = model.x_dot,
            x_dot = -model.x,
        ),
    )
end

function my_updates_fcn(t, model)
    println(model.io, "$(float(t)), $(model.x)")
    return UpdatesOutput()
end

@testset "OutputFileDescription" begin

    # Test that it can open a file for us.
    history, _, m = simulate(
        nothing;
        t = 0 : 0.1 : 1,
        init_fcn = my_init_fcn,
        rates_fcn = my_rates_fcn,
        updates_fcn = my_updates_fcn,
        options = SimOptions(;
            outdir = joinpath(out_dir, "files"),
        ),
    )

    # Test that it put the file there, we wrote to it, and now it's closed.
    out_file_name = joinpath(out_dir, "files", "my_file.txt")
    @test isfile(out_file_name) == true
    @test isopen(m.io) == false
    open(out_file_name) do f
        for (t, x) in zip(history["/"]["x"].time, history["/"]["x"].data)
            if t != 0 # We only write on updates.
                @test readline(f) == "$(float(t)), $x"
            end
        end
    end

end

# This tests structured file outputs.
@kwdef struct MyModel2
    x::Float64
    x_dot::Float64
    io::Tuple{String, IOStream}
end

# We'll specify custom open/close functions.
function my_open_fcn(file_name)
    f = open(file_name, "w")
    println(f, "Time, State")
    return (file_name, f)
end
function my_close_fcn(artifacts_from_open_fcn)
    file_name, f = artifacts_from_open_fcn
    println(f, "eof")
    close(f)
    return nothing
end

function my_init_fcn2(t, params, seed)
    return ModelDescription(;
        type = MyModel2,
        continuous_states = (;
            x = 1.,
            x_dot = 0.,
        ),
        files = (;
            io = OutputFileDescription(;
                name = "my_file.txt",
                open_fcn = my_open_fcn,
                close_fcn = my_close_fcn,
            ),
        ),
    )
end

function my_rates_fcn2(t, model)
    return RatesOutput(;
        rates = (;
            x = model.x_dot,
            x_dot = -model.x,
        ),
    )
end

function my_updates_fcn2(t, model)
    println(model.io[2], "$(float(t)), $(model.x)")
    return UpdatesOutput()
end

@testset "Customization of OutputFileDescription" begin

    # Test that it can open a file for us. Here, we'll put the whole thing in another model
    # to test that file scope is correct.
    history, _, m = simulate(
        nothing;
        t = 0 : 0.3 : 1,
        init_fcn = (t, params, seed) -> ModelDescription(;
            models = (;
                my_model = my_init_fcn2(t, nothing, seed / "my_model"),
            ),
        ),
        rates_fcn = (t, model) -> RatesOutput(;
            models = (;
                my_model = my_rates_fcn2(t, model.my_model),
            ),
        ),
        updates_fcn = (t, model) -> UpdatesOutput(;
            models = (;
                my_model = my_updates_fcn2(t, model.my_model),
            ),
        ),
        options = SimOptions(;
            outdir = joinpath(out_dir, "files"),
        ),
    )

    # Test that it put the file there, we wrote to it, and now it's closed.
    out_file_name = joinpath(out_dir, "files", "my_model", "my_file.txt")
    @test isfile(out_file_name) == true
    @test isopen(m.my_model.io[2]) == false
    open(out_file_name) do f
        for (t, x) in zip(history["/my_model"]["x"].time, history["/my_model"]["x"].data)
            if t == 0
                readline(f) # Burn off the header row.
            else # We only write on updates.
                @test readline(f) == "$(float(t)), $x"
            end
        end
    end

end


function my_init_fcn3(t, params, seed; name, scoped)
    return ModelDescription(;
        type = MyModel,
        continuous_states = (;
            x = 1.,
            x_dot = 0.,
        ),
        files = (;
            io = OutputFileDescription(;
                name,
                scoped,
            ),
        ),
    )
end

@testset "initialization" for scoped in (true, false), output in (false, true)

    temp_dir = joinpath(out_dir, "files", "temp")
    mkpath(temp_dir)
    cd(temp_dir) do

        # Run this both scoped and unscoped and with and without an output directory.
        initialize(
            nothing;
            init_fcn = (args...) -> ModelDescription(;
                models = (;
                    my_other_model = ModelDescription(;
                        type = MyModel,
                        continuous_states = (;
                            x = 1.,
                            x_dot = 0.,
                        ),
                        files = (;
                            io = OutputFileDescription(;
                                name = "my_file_" * (scoped ? "" : "un") * "scoped.txt",
                                scoped,
                            ),
                        ),
                    ),
                ),
            ),
            outdir = output ? joinpath(out_dir, "files") : nothing,
        ) do m
            if output # Files should be in out_dir/files.
                @test isopen(m.my_other_model.io)
                if scoped
                    @test isfile(joinpath(out_dir, "files", "my_other_model", "my_file_scoped.txt"))
                else
                    @test isfile(joinpath(out_dir, "files", "my_file_unscoped.txt"))
                end
            else # Files should be local.
                if scoped
                    @test isfile(joinpath(temp_dir, "my_other_model", "my_file_scoped.txt"))
                else
                    @test isfile(joinpath(temp_dir, "my_file_unscoped.txt"))
                end
            end
        end

    end

end

end
