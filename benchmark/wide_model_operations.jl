# Run this like so:
#
#  julia --project=. benchmark/wide_model_operations.jl
#
module BenchmarkWideModelOperations

import SystemsOfSystems
using SystemsOfSystems: ModelStateDescription, RatesOutput

const ContinuousProblems = SystemsOfSystems.ContinuousProblems

####################
# Benchmark Inputs #
####################

function make_propagation_inputs(::Val{N}) where {N}

    names = ntuple(index -> Symbol("model_$index"), Val(N))
    submodels = NamedTuple{names}(
        ntuple(Val(N)) do index
            ModelStateDescription{Nothing}(;
                continuous_states = (; x = Float64(index)),
            )
        end,
    )

    function make_stage(multiplier)
        return NamedTuple{names}(
            ntuple(Val(N)) do index
                RatesOutput(; rates = (; x = multiplier * index))
            end,
        )
    end

    return (;
        submodels,
        gains = (1/4, 1/2),
        model_rates_at_stages = (make_stage(1.), make_stage(2.)),
    )

end

function make_stop_input(::Val{N}, stop_last) where {N}
    names = ntuple(index -> Symbol("model_$index"), Val(N))
    models = NamedTuple{names}(
        ntuple(Val(N)) do index
            RatesOutput(; stop = stop_last && index == N)
        end,
    )
    return RatesOutput(; models)
end

#########################
# Measurement Utilities #
#########################

function propagate_models(inputs)
    return ContinuousProblems.propagate_models(
        inputs.submodels,
        inputs.gains,
        inputs.model_rates_at_stages,
    )
end

find_model_requested_stop(output) = SystemsOfSystems.find_model_requested_stop(output)

# Keeping the repetition loop out of the measurement caller prevents constant propagation
# of the benchmark inputs from replacing the work that we intend to measure.
@noinline function run_many(f, input, iterations)
    result = nothing
    for _ in 1:iterations
        result = f(input)
    end
    return result
end

function measure(label, f, input, iterations)

    # Compile all methods before measuring, then collect any compilation garbage so it does
    # not influence the measured garbage-collection time.
    run_many(f, input, 1)
    GC.gc()

    elapsed = @elapsed result = run_many(f, input, iterations)
    allocated = @allocated allocation_result = run_many(f, input, iterations)
    @assert typeof(result) == typeof(allocation_result)

    println(
        rpad(label, 18),
        lpad(round(1e6 * elapsed / iterations; digits = 3), 10),
        " μs/call, ",
        lpad(round(allocated / iterations; digits = 1), 10),
        " bytes/call",
    )
    return result

end

#########################
# Wide Model Operations #
#########################

function benchmark_width(width::Val{N}) where {N}

    println("width = $N")

    # Propagation returns a new mutable ModelStateDescription for every submodel, so those
    # necessary result allocations are included in the reported allocation count.
    propagation_inputs = make_propagation_inputs(width)
    propagated = measure(
        "propagation",
        propagate_models,
        propagation_inputs,
        10_000,
    )
    for index in 1:N
        expected = 2.25 * index
        @assert propagated[index].continuous_states.x == expected
    end

    # The no-stop case traverses every submodel without creating a stop path. The last-stop
    # case also traverses the whole tuple, then includes the unavoidable path construction.
    no_stop_input = make_stop_input(width, false)
    no_stop = measure(
        "stop (none)",
        find_model_requested_stop,
        no_stop_input,
        100_000,
    )
    @assert isnothing(no_stop)

    last_stop_input = make_stop_input(width, true)
    last_stop = measure(
        "stop (last)",
        find_model_requested_stop,
        last_stop_input,
        100_000,
    )
    @assert last_stop isa SystemsOfSystems.ModelRequestedStop
    @assert last_stop.model_path == "/models/model_$N"

    println()

end

function run_benchmarks()
    for width in (Val(5), Val(20), Val(64))
        benchmark_width(width)
    end
end

end # BenchmarkWideModelOperations

BenchmarkWideModelOperations.run_benchmarks()
