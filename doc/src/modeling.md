# Modeling

This describes how to build a model using SystemsOfSystems.

## Primary Function Outputs

### Initialization

`ModelDescription`

... models must be named tuples... practical limitations on the number sub-models a model can have...

### Continuous-Time Dynamics

`RatesOutput`

### Discrete-Time Dynamics

`UpdatesOutput`

## Continuous States

... change via their derivatives...

... can be updated discretely too...

## Discrete States

## Random Variables

### Continuous Random Variables

... can be any type that is callable with (rng, t_last, t_next)...

ContinuousWhiteNoise is an example...

### Discrete Random Variables

... can be any type that is callable with (rng, t)...

DiscreteWhiteNoise is an example...

## Schedules

examples of the types available today

## Resources

examples of the types available today

## Custom `t_next`

notes on using `t_next` without a schedule
