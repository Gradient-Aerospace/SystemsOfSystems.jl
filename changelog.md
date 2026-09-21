# SystemsOfSystems Change Log

## Unreleased

### Non-Breaking

* Updated compat bounds for OrderedCollections to include 2.0
* Added iteration over TimeSeries
* Added `AlwaysTriggeringSchedule`
* Added `Interrupted` stop reason for interruptions like the `InterruptException` resulting from ctrl+c. This is not considered a failure, and `succeeded(history)` will return true for interrupted runs.

## v1.0.0

* Initial release
