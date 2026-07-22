module TestModelFilters

using Test
using SystemsOfSystems
using SystemsOfSystems: ModelFilters, Samplers

@testset "compact regular-expression model filters" begin

    regular_sampler = Samplers.RegularSampler(1//10)
    compact_entries = [
        r"^/$" => regular_sampler,
        r"^/" => Samplers.CompleteSampler(),
    ]

    # A pair converts directly to the explicit entry type, allowing ordinary typed-array
    # construction to use the compact syntax too.
    converted_entry = convert(ModelFilters.RegexModelEntry, first(compact_entries))
    @test converted_entry.expression == r"^/$"
    @test converted_entry.sampler === regular_sampler

    # Both positional and keyword filter constructors accept the compact vector. Entry
    # order remains significant because the first matching regular expression wins.
    positional_filter = ModelFilters.RegexModelFilter(compact_entries)
    keyword_filter = ModelFilters.RegexModelFilter(; rules = compact_entries)
    for filter in (positional_filter, keyword_filter)
        @test length(filter.rules) == 2
        @test ModelFilters.get_model_sampler(filter, "/") === regular_sampler
        @test ModelFilters.get_model_sampler(filter, "/models/plant") isa
            Samplers.CompleteSampler
        @test ModelFilters.get_model_sampler(filter, "no-leading-slash") isa
            Samplers.NullSampler
    end

    # The original explicit construction remains available and produces the same matching
    # behavior as the compact form.
    explicit_filter = ModelFilters.RegexModelFilter(;
        rules = [
            ModelFilters.RegexModelEntry(;
                expression = r"^/$",
                sampler = regular_sampler,
            ),
            ModelFilters.RegexModelEntry(;
                expression = r"^/",
                sampler = Samplers.CompleteSampler(),
            ),
        ],
    )
    @test ModelFilters.get_model_sampler(explicit_filter, "/") === regular_sampler
    @test ModelFilters.get_model_sampler(explicit_filter, "/models/plant") isa
        Samplers.CompleteSampler

end

end # TestModelFilters
