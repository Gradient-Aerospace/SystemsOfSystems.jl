# Initialization

It's often useful to create a model outside of the context of simulation, such as when developing the model or harnessing the models for analysis purposes. The `initilize` function provides this behavior.

The preferred method is the `do` form. It works like this:

```julia
initialize(user_data; init_fcn = ...) do my_model

    # Do something with my_model.

end
```

This is particularly helpful when the model uses [Resources](@ref), because this makes sure the resources are closed when the model is no longer necessary.

Further, this works with a model description directly, in which case no `init_fcn` is necessary:

```julia
initialize(model_description) do my_model

    # Do something with my_model.

end
```

Using the `do` form isn't required. Two other methods exist:

```julia
my_model = initialize(user_data; init_fcn...)
my_model = initialize(model_description)
```

See below for the full interface.

```@docs
SystemsOfSystems.initialize
```
