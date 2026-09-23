# Reading HDF5 Results Outside Julia

Simulation histories and logs are intended to be useful in ordinary HDF5 tools outside Julia. This page defines the public groups, datasets, and attributes that external readers can rely on. Additional entries may be present for Julia reconstruction or other implementation needs. Readers should use the documented entries and tolerate extra entries, rather than require an exact set of keys.

SystemsOfSystems defines the structure that identifies runs, models, variables, and their metadata. Sample values use the documented [HDF5Vectors storage format](https://gradient-aerospace.github.io/HDF5Vectors.jl/stable/storage_layout/). Its portable data and schema entries are also available to external readers; they are not Julia implementation details. The distinction is between documented, externally interpretable data and entries needed only to reconstruct Julia objects.

## Format Versions and Compatibility

History metadata and logs have independent integer format versions because either can be used without the other. These describe the stored layout, not the SystemsOfSystems package release. HDF5Vectors versions its nested value encodings separately.

| Stored object | Format attribute | Versions currently read | Version written for new data |
| --- | --- | --- | --- |
| History metadata group | `sim_history_version` | `1` | `1` |
| Log group, including a NullLog | `log_format_version` | `1`, plus the existing unversioned legacy layout | `1` |

Each loader checks its group's version before interpreting its contents or deserializing Julia objects. A missing history version, a malformed version, or an unsupported explicit version produces an `ArgumentError` identifying the group and the problem. Unversioned logs are the one deliberate exception: the existing legacy log reader handles them, whether stored at the file root or in a selected group. Missing versions do not provide a general way to infer future layouts.

Newly written history and log groups also carry a string attribute, `systems_of_systems_version`, identifying the writer's package release. This is provenance, not a compatibility gate. It is optional when reading, including for files written before provenance was recorded. A newer writer package version does not itself make a supported format unreadable.

Saving an in-memory log writes the current log format. Copying an existing HDF5 log preserves its format version and writer provenance, including absent attributes on a legacy log. The same applies when saving history metadata beside an existing log. Neither operation converts the log to a newer format; new history metadata records its own current version and writer independently.

### Support Policy

SystemsOfSystems uses independent format versions with bounded support. Within a package major release, newer releases retain the supported-format readers provided by earlier releases in that major. A future package major release may drop support for older formats; permanent backward reading is not promised. The support table above and release notes identify supported formats and any removals. A package major release need not change a format version if the storage layout remains compatible.

Format versions change when the interpretation of public data changes incompatibly. Adding optional entries that readers can ignore, or changing compression and chunk sizes, does not require a new format version. Renaming or removing a public entry, changing its meaning, or changing its representation incompatibly does. Such changes to the existing public writer interface belong in a package major release. Configurable group locations are already part of the format and do not require a version change.

Support applies to the documented layout and its portable data. It does not guarantee reconstruction of arbitrary application types or compatibility of Julia-serialized objects across Julia or dependency versions. Changes to internal reconstruction entries do not automatically require a public format-version change, but they still need to respect the package's supported loading behavior. Nested HDF5Vectors encodings have their own compatibility requirements.

## Locating a Run and Its Log

By default, a saved history has two groups:

```text
/history/                  # Run metadata
/log/                      # Root model's logged data
```

These names are defaults, not mandatory locations. A caller may place either group elsewhere. An external reader starts from the chosen history group and follows its `log_path` dataset to the log. A standalone log has no history group; its location is supplied directly, with `/log` as the default for new files.

The following entries are relative to the history group:

| Entry | Representation | Meaning |
| --- | --- | --- |
| `sim_history_version` | Integer **attribute**, currently `1` | Version of the history metadata format. |
| `systems_of_systems_version` | Optional string **attribute** | Writer package version, for provenance only. |
| `log_path` | Scalar string dataset | Absolute HDF5 path to the log group in the same file. |
| `t_start`, `t_stop` | Scalar `Float64` datasets | Requested start time and last completed simulation time, using the simulation's time coordinate. |
| `stop/type` | Scalar string dataset | Original termination reason's type name, for identification when inspecting results. |
| `stop/description` | Scalar string dataset | Human-readable reason for termination. |
| `stop/is_failure` | Scalar Boolean dataset | Whether the run terminated with a failure reason. A normal early stop is not a failure. |
| `stop/details` | Optional scalar string dataset | Additional diagnostic text, such as an exception and stack trace. Absence means no additional text was saved. |
| `stop/value/` | HDF5Vectors group | Structured saved termination reason. Its schema describes the recorded fields; it may represent a descriptive record rather than the original reason. |
| `model/` | Optional HDF5Vectors group | One final model value, present when saved with `save_model = true`. Its schema and portability depend on the application's model type. |

All entries except `stop/details`, `model`, and the provenance attribute are required in a saved history. Type names and descriptions are descriptive text: their wording is not a machine-readable classification scheme. `stop/is_failure` supplies the success/failure distinction without parsing either string.

Because `log_path` is within the file, moving or renaming the file preserves it. Moving a log group within the file requires updating any history that refers to it. A log group with `is_null = true` records that logging was disabled; it has no model tree. Ordinary logs need not have an `is_null` entry.

## Models and Variables

The log group carries the integer attribute `log_format_version` and, for newly written logs, the string provenance attribute `systems_of_systems_version`. These attributes describe the whole log and are not repeated on child models. Legacy unversioned logs may lack both attributes.

For an ordinary log, this group also represents the root model. Each model has the following public entries, relative to its own group:

| Entry | Representation | Meaning |
| --- | --- | --- |
| `type` | Scalar string dataset | Descriptive name of the model's type. |
| `names/models` | String vector dataset | Child model names in their original order. |
| `names/constants` | String vector dataset | Names of successfully saved constants, in order. |
| `names/continuous_states` | String vector dataset | Names of logged continuous states, in order. |
| `names/discrete_states` | String vector dataset | Names of logged discrete states, in order. |
| `names/continuous_outputs` | String vector dataset | Names of logged continuous outputs, in order. |
| `names/discrete_outputs` | String vector dataset | Names of logged discrete outputs, in order. |
| `models/<name>/` | Group per child model | The same model layout, recursively. |
| `constants/<name>/` | Group per saved constant | Constant value and metadata, described below. |
| `timeseries/<name>/` | Group per logged state or output | Time-series samples and metadata, described below. |

The `names` datasets are present even when their lists are empty. `models` and `timeseries` may be absent when there are no corresponding entries. Readers should follow the name lists rather than infer model order or variable categories from HDF5 group iteration. Logging policies can omit variables, and unsupported constants are omitted with a warning when saving.

For example, the time series `position` belonging to model `/vehicle/controller` is at:

```text
/log/models/vehicle/models/controller/timeseries/position/
```

If the log lives at `/runs/first/samples`, the same relative path starts there instead. Model paths used by the simulation do not include the storage group names or the repeated `models` components.

## Time Series

Each `timeseries/<name>` group exposes these entries:

| Entry | Representation | Meaning |
| --- | --- | --- |
| `title` | Scalar string dataset | Display title. |
| `path` | Scalar string dataset | Variable path within the model hierarchy, such as `/vehicle/position`; not an HDF5 storage path. |
| `discrete` | Scalar Boolean dataset | Whether the series is discrete rather than continuous. |
| `time_label`, `time_units` | Scalar string datasets | Label and units of the time coordinate. |
| `labels`, `units` | String vector datasets | Corresponding labels and units for the scalar dimensions of a sample. |
| `groups_are_missing` | Scalar Boolean dataset | Whether dimension grouping was unspecified. |
| `group_labels` | String vector dataset | Names of dimension groups, in order. |
| `group_dimension_counts` | Integer vector dataset | Number of dimension labels in each group. |
| `group_dimension_labels` | String vector dataset | Concatenated dimension-label lists for all groups. |
| `interpolator_type` | Scalar string dataset | Descriptive name of the interpolator used in Julia. |
| `time/` | HDF5Vectors group | Recorded sample times, with `Float64` elements. |
| `data/` | HDF5Vectors group | Corresponding recorded values. |

The time and data vectors have the same logical length for a successfully recorded series. Different variables may have different sample times; each series has its own `time` vector. The times are actual logged samples and need not include the history's final `t_stop`.

Dimension group membership is reconstructed by splitting `group_dimension_labels` according to `group_dimension_counts`, in the order of `group_labels`. For example, counts `[2, 1]` divide three labels into a two-dimension group and a one-dimension group. When `groups_are_missing` is true, grouping was unspecified; this differs from an explicitly empty grouping.

The interpolator's name is useful when inspecting the series, but does not encode all parameters or define how another language should execute a custom interpolator. Reading the recorded samples does not require reconstructing it.

### Sample Storage

`time` and `data` are vector groups, not necessarily datasets. For the HDF5Vectors version 1 format, each contains public `metadata/count`, `metadata/format_name`, `metadata/format_version`, and `metadata/schema` entries describing its length and encoding. The physical values are beneath its own `data` group. This accounts for the repeated `data` component in the paths below:

```text
timeseries/position/
├── time/
│   ├── metadata/ ...
│   └── data/values             # Float64 sample times
└── data/
    ├── metadata/ ...
    └── data/values             # Scalar or dense-array samples
```

For scalar numbers, `data/data/values` is a one-dimensional dataset. For a fixed-size vector sample with three components, the raw HDF5 dataset has shape `(sample_count, 3)`. HDF5.jl reverses dimensions at the HDF5 C boundary: matrix samples of Julia size `(2, 3)` have raw HDF5 shape `(sample_count, 3, 2)`. This is an array-ordering convention, not a transpose applied to the model's values. The HDF5Vectors storage documentation describes how these extents relate to the Julia view.

Portable record samples use field names beneath `data/data` instead of a single `values` dataset. For example, a record with a scalar field `temperature` has a column at `data/data/temperature/values`; nested records repeat their field names. Readers can inspect `data/metadata/schema/kind` and its child schemas to determine the representation, rather than assume every variable uses `data/data/values`. The HDF5Vectors documentation defines the complete scalar, dense, record, and other layouts.

Some values, such as dynamically sized arrays without declared dimensions, may be encoded with Julia serialization. Their time series and metadata are still discoverable, but their payload cannot be reconstructed by an ordinary HDF5 reader. The public layout does not imply that every possible Julia value has a portable encoding. Applications that need cross-language access should select data types and storage options with portable HDF5Vectors representations.

## Constants

Each `constants/<name>` group contains a `value` HDF5Vectors group and ordinary metadata datasets:

| Entry | Representation | Meaning |
| --- | --- | --- |
| `value_is_missing` | Scalar Boolean dataset | Whether the constant's value is missing. |
| `value/` | HDF5Vectors group | The saved value; normally one element. If `value_is_missing` is true, the payload should not be interpreted as a present value. |
| `title` | Scalar string dataset | Display title or variable path. |
| `labels`, `units` | String vector datasets | Dimension labels and units; empty when none were supplied for an ordinary constant. |
| `value_type` | Optional scalar string dataset | Descriptive value type from a `VariableDescription`. |

Constants supplied with `VariableDescription` also have the grouping and `interpolator_type` entries described for time series. Ordinary constants need not have those entries. External readers can inspect the value's HDF5Vectors schema without using `value_type` to reconstruct a Julia type.

## Julia Reconstruction Entries

The public entries above are sufficient to navigate a run and read its portably encoded log data. Other entries are not part of the SystemsOfSystems external-reader contract. In particular, `serialized_type`, `serialized_value_type`, and `serialized_interpolator` store Julia reconstruction information. The `is_variable_description` flag selects a Julia wrapper when loading a constant. These entries may change without changing the public layout.

The public HDF5Vectors groups, including `stop/value` and the optional final `model`, can be inspected according to their ordinary stored schemas. SystemsOfSystems does not promise one fixed set of fields for arbitrary application models or all termination reasons: the schema describes the value actually saved. The readable `stop` datasets provide a uniform termination summary without interpreting a particular reason's structure. Julia-serialized schemas and payloads remain Julia-specific, even when stored inside an otherwise public group.

Adding entries does not invalidate this layout. Existing public paths and their meanings are compatibility commitments, while unlisted entries may change. Readers should also honor the separate HDF5Vectors format version before interpreting vector storage. This page describes newly written files; older standalone logs may use a root-level model layout or lack metadata introduced in later releases.
