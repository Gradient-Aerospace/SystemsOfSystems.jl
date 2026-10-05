module CleanupErrors

export AbstractCleanupError, CleanupError, CleanupErrorSummary, cleanup_context, cleanup_details

"""
A failure while closing a hook or resource. Custom errors implement `cleanup_context`
and `cleanup_details`, returning strings that identify the context and describe the failure.
"""
abstract type AbstractCleanupError end

"""
    cleanup_context(err::AbstractCleanupError)

Returns a string identifying the hook or resource that failed to close.
"""
function cleanup_context end

"""
    cleanup_details(err::AbstractCleanupError)

Returns a string containing diagnostic information about the cleanup failure.
For a `CleanupError`, this includes the rendered exception and stack trace.
"""
function cleanup_details end

"""
    CleanupError(context, exception, trace)

A cleanup failure from this process, retaining the thrown value and stack trace.
"""
struct CleanupError <: AbstractCleanupError
    context::String
    exception::Any
    trace::Any
end
cleanup_context(err::CleanupError) = err.context
cleanup_details(err::CleanupError) = sprint(showerror, err.exception, err.trace)

"""
    CleanupErrorSummary(err::AbstractCleanupError)
    CleanupErrorSummary(; type = "", context = "", details = "")

Stores the cleanup-error API results as portable text, without retaining exception objects
or stack traces. Constructing a summary from an existing summary returns it unchanged.

Fields:

* `type::String`: The original cleanup error's type name, for identification.
* `context::String`: The result of `cleanup_context` on the original error.
* `details::String`: The result of `cleanup_details` on the original error.
"""
@kwdef struct CleanupErrorSummary <: AbstractCleanupError
    type::String = ""
    context::String = ""
    details::String = ""
end
cleanup_context(err::CleanupErrorSummary) = err.context
cleanup_details(err::CleanupErrorSummary) = err.details

function CleanupErrorSummary(err::AbstractCleanupError)
    return CleanupErrorSummary(;
        type = string(typeof(err)),
        context = cleanup_context(err),
        details = cleanup_details(err),
    )
end
CleanupErrorSummary(err::CleanupErrorSummary) = err

end # module CleanupErrors
