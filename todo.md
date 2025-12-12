# TODOs

## General

- [ ] Rename "monitors" to "hooks"?

## Documentation

- [ ] Tidy up the public interface and fill in docstrings.
- [ ] Create a tutorial.

## Testing

- [x] Test both solvers.
- [x] Test all logs.
- [ ] Test systems of systems.
- [ ] Test random variables.
- [ ] Test all the types we intend to support.
- [ ] Test with VariableDescriptions.
- [ ] Test monitor.
- [ ] Test plots.
- [ ] Test very long sims to make sure steps keep working as expected.
- [ ] Test for type stability.
- [ ] Test documentation with jldoctest.

## Features

- [ ] Support `missing` outputs (don't log them).
- [ ] Allow submodels to be tuples or vectors.
- [ ] Create general ButcherTableau solvers (fixed step, adaptive step) and have RungeKutta4 and DormandPrince54 use it. Add a couple of other solvers, like Bogacki-Shampine 3(2) and Heun.
