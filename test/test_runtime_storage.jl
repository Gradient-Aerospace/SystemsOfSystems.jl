module RuntimeStorageProcessTests

using Test

# These cases deliberately add methods to supported adapters. Separate processes keep
# those extensions local to their own compatibility test and preserve active-path checks.
@testset "Private runtime storage ownership and extension boundaries" begin

    project_file = Base.active_project()
    @assert project_file !== nothing
    project = dirname(project_file)
    directory = joinpath(@__DIR__, "runtime_storage")
    cases = (
        ("state_ownership.jl", ()),
        ("stage_ownership.jl", ()),
        ("adaptive_ownership.jl", ()),
        ("late_arithmetic_extensions.jl", ("weighted",)),
        ("late_arithmetic_extensions.jl", ("variable",)),
    )
    for (file, arguments) in cases
        @testset "$file $(join(arguments, ' '))" begin
            path = joinpath(directory, file)
            options = ("--startup-file=no", "--threads=1", "--project=$project")
            command = `$(Base.julia_cmd()) $options $path $arguments`
            @test success(command)
        end
    end

end

end # RuntimeStorageProcessTests
