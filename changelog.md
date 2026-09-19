# SystemsOfSystems Change Log

## Unreleased

### Non-Breaking

* Added `Interrupted` stop reason for interruptions like the `InterruptException` resulting from ctrl+c. This is not considered a failure, and `succeeded(history)` will return true for interrupted runs.

## v1.0.0

* Initial release
