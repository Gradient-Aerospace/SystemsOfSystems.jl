# SystemsOfSystems Change Log

## v1.1.0

### Non-Breaking

* Added iteration over TimeSeries
* Added `AlwaysTriggeringSchedule`
* Added `Interrupted` stop reason for interruptions like the `InterruptException` resulting from ctrl+c. This is not considered a failure, and `succeeded(history)` will return true for interrupted runs.

### Patch

* Updated compat bounds for OrderedCollections to include 2.0

## v1.0.0

* Initial release
