module RunTests

for filename in sort(readdir(@__DIR__))
    if startswith(filename, "test_") && endswith(filename, ".jl")
        include(filename)
    end
end

end # RunTests
