module Resources

export AbstractResource, OutputFile, Resource, open_resource, close_resource

"""
    AbstractResource

An abstract type for managing external resources for a model. All subtypes are expected to
implement `open_resource` and `close_resource`.
"""
abstract type AbstractResource end

"""
    open_resource(resource::AbstractResource, outdir, model_path)

Opens a `resource` using the top-level `outdir` provided to the `simulate` function and the
`model_path` (e.g., `/submodel1/subsubmodel2`), returning a payload that the model will
store.
"""
function open_resource end

"""
    close_resource(::AbstractResource, payload)

Closes a `resource`, using the `payload` provided from the `open_resource` function.
"""
function close_resource end

"""
    Resource

A container for a general resource, like a TCP/IP connection or a shared library.

Fields:

* `open_args` - A vector of arguments to pass to the `open_fcn`
* `open_fcn` - A function to call to open the resource, with the above arguments as inputs,
  returning a "payload" that the model will store during simulation
* `close_fcn` - A function to call to close the resource, with the payload as the input
"""
@kwdef struct Resource <: AbstractResource
    open_args::Vector{Any}
    open_fcn::Any
    close_fcn::Any
end
export Resource

function open_resource(file_desc::Resource, outdir, model_path)
    return file_desc.open_fcn(file_desc.open_args...)
end

function close_resource(desc::Resource, payload)
    desc.close_fcn(payload)
    return nothing
end

"""
    OutputFile

Describes an output file that SystemsOfSystems should open after initialzation and close
after simulation.

After `init_fcn` has run, this will create the requested file `name`. If `name` is an
absolute path, that file will be created. If it is a relative path, the file will be stored
in the `outdir` provided to the `simulate` as `<outdir>/model/submodel/subsubmodel/<name>`
when `scoped == true` and `<outdir>/<name>` otherwise.
"""
@kwdef struct OutputFile <: AbstractResource
    name::String
    scoped::Bool = true
end
export OutputFile

function open_resource(file_desc::OutputFile, outdir, model_path)

    # Figure out where to put the file.
    file_name = file_desc.name
    if !isabspath(file_name)

        # Prepend the model path.
        if file_desc.scoped
            file_name = joinpath(model_path, file_name)
        end

        # The top-level model's path will be "". The first level of submodels will be
        # "/submodel". We need to remove that initial "/" to treat this as a relative path.
        if startswith(file_name, "/")
            file_name = file_name[2:end]
        end

        # Prepend the outdir.
        if !isnothing(outdir)
            file_name = joinpath(outdir, file_name)
        end

    end

    # Make sure the whole path up to that file exists.
    file_dir = dirname(file_name)
    mkpath(file_dir)

    # Call the user's function to let it open the file or do whatever it does, and store
    # whatever it returns as the "file".
    return open(file_name, "w")

end

function close_resource(desc::OutputFile, file_handle)
    close(file_handle)
    return nothing
end

end
