module TestStepTriggering

using Test
using SystemsOfSystems

@testset "is_regular_step_triggering" begin
    @test is_regular_step_triggering(10.1, 0.05) == true
    @test is_regular_step_triggering(10.1, 0.20) == false
    @test is_regular_step_triggering(10.1, 0.) == true # 0 means "always triggering"
    @test is_regular_step_triggering(10.1, 0.20, 0.1) == true
    @test is_regular_step_triggering(10.1, 1., 0.0) == false
    @test is_regular_step_triggering(10.1, 1., 0.1) == true
end

end # TestStepTriggering

