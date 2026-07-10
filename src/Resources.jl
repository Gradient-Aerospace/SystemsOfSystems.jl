module Resources

export AbstractResource, OutputFile, Resource, ResourceInputs, open_resource, close_resource

"""
    ResourceInputs

Stores a set of inputs that an `AbstractResource` can use in its `open_resource` method.

Fields:

* `outdir`: The top-level output directory given to the `simulate` function
* `model_path`: The "path" to the model, the same as the keys used to access model results
  in a `SimHistory`

This structure is not fixed. Fields may be added to this in the future, but the above fields
will remain (or the release will be marked as a breaking change).
"""
@kwdef struct ResourceInputs
    outdir::Union{Nothing, String}
    model_path::String
end

"""
    AbstractResource

An abstract type for managing external resources for a model. All subtypes are expected to
implement `open_resource` and `close_resource`.
"""
abstract type AbstractResource end

"""
    open_resource(resource::AbstractResource, inputs::ResourceInputs)

Opens a `resource` using the given `input` (see `ResourceInputs`).
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

* `open_args` - A tuple of arguments to pass to the `open_fcn`
* `open_fcn` - A function to call to open the resource. The first argument will be a
  `ResourceInputs`, and the remaining arguments will be the `open_args`. This should return
  a "payload" that the model will store during simulation.
* `close_fcn` - A function to call to close the resource, with the payload as the input
"""
@kwdef struct Resource <: AbstractResource
    open_args::Tuple
    open_fcn::Any
    close_fcn::Any
end
export Resource

function open_resource(file_desc::Resource, inputs::ResourceInputs)
    return file_desc.open_fcn(inputs, file_desc.open_args...)
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

function open_resource(file_desc::OutputFile, inputs::ResourceInputs)

    # Figure out where to put the file.
    file_name = file_desc.name
    if !isabspath(file_name)

        # Prepend the model path.
        if file_desc.scoped
            file_name = joinpath(inputs.model_path, file_name)
        end

        # The top-level model's path will be "". The first level of submodels will be
        # "/submodel". We need to remove that initial "/" to treat this as a relative path.
        if startswith(file_name, "/")
            file_name = file_name[2:end]
        end

        # Prepend the outdir.
        if !isnothing(inputs.outdir)
            file_name = joinpath(inputs.outdir, file_name)
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
