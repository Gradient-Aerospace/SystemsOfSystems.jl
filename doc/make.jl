using Documenter, SystemsOfSystems

makedocs(;
    sitename = "SystemsOfSystems Documentation",
    pages = [
        "Introduction" => "index.md",
        "Control System Example" => "control_system_example.md",
        "Simulation" => "simulation.md",
        "Modeling" => "modeling.md",
        "Options" => "options.md",
        "Initialization" => "initialization.md",
        "Time" => "time.md",
    ],
)

# To deploy the docs to GitHub Pages:
# deploydocs(
#     repo = "github.com/Gradient-Aerospace/HDF5Vectors.jl.git",
# )
