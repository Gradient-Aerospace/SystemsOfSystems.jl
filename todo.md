# TODOs

## General

- [ ] Rename "monitors" to "hooks"?

## Documentation

- [ ] Tidy up the public interface and fill in docstrings.
- [ ] Create a tutorial.

## Testing

- [x] Test both solvers.
- [x] Test all logs.
- [x] Test continuous-only sims.
- [x] Test discrete-only sims.
- [x] Test hybrid sims.
- [x] Test without VariableDescriptions.
- [x] Test with VariableDescriptions.
- [x] Test monitor.
- [x] Test systems of systems.
- [x] Test the close function.
- [ ] Test random variables.
- [ ] Test all the types we intend to support.
- [ ] Test plots.
- [ ] Test very long sims to make sure steps keep working as expected.
- [ ] Test for type stability.
- [ ] Test documentation with jldoctest.
- [ ] Test non-zero start times.
- [ ] Figure out how to log the RNG state in a way we could load later.
- [ ] Figure out how to capture console output.

## Features

- [ ] Allow the discrete update to change continuous states too.
- [ ] Support `missing` outputs (don't log them).
- [ ] Allow submodels to be tuples or vectors.
- [ ] Create general ButcherTableau solvers (fixed step, adaptive step) and have RungeKutta4 and DormandPrince54 use it. Add a couple of other solvers, like Bogacki-Shampine 3(2) and Heun.
- [ ] Allow each model to provide its own function to return the integration error of its states. This would be useful for, e.g., specifying that a position vector's error is the norm of the whole thing, not the individual elements.