# History metadata and logs can be read independently, so each has its own format version.
# These numbers describe storage layouts, not package releases. The current readers also
# accept the existing unversioned log layout, but histories have always carried a version.
const history_format_version = 1
const log_format_version = 1

function write_format_version(group, attribute, version)
    HDF5.attrs(group)[attribute] = version
    HDF5.attrs(group)["systems_of_systems_version"] = string(Base.pkgversion(SystemsOfSystems))
    return nothing
end

function check_format_version(group, attribute, supported_version; allow_unversioned = false)

    # A missing log version selects the legacy reader already supported by this extension.
    # An explicit version must never fall back to that reader: an unfamiliar layout may
    # have different meanings even when its dataset names happen to look familiar.
    attributes = HDF5.attrs(group)
    location = "$(HDF5.filename(group)):$(HDF5.name(group))"
    if !haskey(attributes, attribute)
        if allow_unversioned
            return nothing
        end
        throw(ArgumentError(
            "Missing $attribute attribute at $location. " *
            "Supported format version: $supported_version.",
        ))
    end

    # Version numbers are scalar integers. In particular, true and 1.0 must not pass as
    # version 1 merely because Julia considers them numerically equal to it.
    version = attributes[attribute]
    if !(version isa Integer) || version isa Bool
        throw(ArgumentError(
            "Invalid $attribute at $location: expected a scalar integer, " *
            "got $(repr(version)).",
        ))
    end
    if version != supported_version
        throw(ArgumentError(
            "Unsupported $attribute = $version at $location. " *
            "Supported format version: $supported_version. " *
            "Use a SystemsOfSystems release that supports the stored format.",
        ))
    end
    return nothing

end
