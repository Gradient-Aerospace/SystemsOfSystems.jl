# SystemsOfSystems Change Log

## Unreleased

### Non-Breaking

* Added `save_sim_history` and `load_sim_history`.
* Added independent HDF5 history and log format-version checks and writer-package provenance. Existing unversioned logs remain readable; copied HDF5 logs retain their original format and provenance.
* Changed default HDF5 group for `save_log_to_hdf5` from "/" to "/log".
* Added `cleanup_errors` to `SimHistory`.
* Added an API for termination reasons (`finished`, `failed`, `interrupted`, `describe`, and `details`).
* Added `gc_policy` option to `SimOptions`.
* Reduced allocations when logging continuous and discrete outputs with different field types.

## v1.1.0

### Non-Breaking

* Added iteration over TimeSeries
* Added `AlwaysTriggeringSchedule`
* Added `Interrupted` stop reason for interruptions like the `InterruptException` resulting from ctrl+c. This is not considered a failure, and `succeeded(history)` will return true for interrupted runs.
### Patch

* Updated compat bounds for OrderedCollections to include 2.0.

### Patch

* Updated compat bounds for OrderedCollections to include 2.0

## v1.0.0

* Initial release
