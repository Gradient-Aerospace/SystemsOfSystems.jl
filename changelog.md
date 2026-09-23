# SystemsOfSystems Change Log

## Unreleased

### Non-Breaking

* Added iteration over TimeSeries.
* Added `AlwaysTriggeringSchedule`.
* Added `Interrupted` stop reason for interruptions like the `InterruptException` resulting from ctrl+c. This is not considered a failure, and `succeeded(history)` will return true for interrupted runs.
* Added `save_sim_history` and `load_sim_history`.
* Added independent HDF5 history and log format-version checks and writer-package provenance. Existing unversioned logs remain readable; copied HDF5 logs retain their original format and provenance.
* Changed default HDF5 group for `save_log_to_hdf5` from "/" to "/log".

### Patch

* Updated compat bounds for OrderedCollections to include 2.0.

## v1.0.0

* Initial release
