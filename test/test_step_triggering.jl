module TestStepTriggering

using Test
using SystemsOfSystems

@testset "is_regular_step_triggering" begin
    @test Schedules.is_regular_step_triggering(10.1, 0.05) == true
    @test Schedules.is_regular_step_triggering(10.1, 0.20) == false
    @test Schedules.is_regular_step_triggering(10.1, 0.) == true
    @test Schedules.is_regular_step_triggering(10.1, 0.20, 0.1) == true
    @test Schedules.is_regular_step_triggering(10.1, 1., 0.0) == false
    @test Schedules.is_regular_step_triggering(10.1, 1., 0.1) == true
end

end # TestStepTriggering
