using Documenter, SystemsOfSystems

makedocs(;
    sitename = "SystemsOfSystems Documentation",
    pages = [
        "Introduction" => "index.md",
        "Control System Example" => "control_system_example.md",
        # "Modeling" => "modeling.md",
        # "Sim Options" => "options.md",
    ],
)

# To deploy the docs to GitHub Pages:
# deploydocs(
#     repo = "github.com/Gradient-Aerospace/HDF5Vectors.jl.git",
# )
