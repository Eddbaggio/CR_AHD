import pytest

import src.cr_ahd.utility_module.plotting as pl
from src.cr_ahd.core_module import solution as slt

SKIP_TEST_PLOTTING = True


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_inst_gh_0(inst_gh_0):
    solution = slt.CAHDSolution(inst_gh_0)
    pl.plot_solution_2(inst_gh_0, solution, show=True)
    pass


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_circular_inst_gen(instance_generator_circular):
    inst = instance_generator_circular()
    solution = slt.CAHDSolution(inst)
    pl.plot_solution_2(inst, solution, show=True)
    pass


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_inst_and_sol_gh_0_9_ass_6_routed(inst_and_sol_gh_0_ass9_routed6):
    inst, solution = inst_and_sol_gh_0_ass9_routed6
    pl.plot_solution_2(inst, solution, show=True)
    pass


@pytest.mark.skipif(False, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_inst_sol_circular_ass_routed_generator(inst_sol_circular_ass_routed_generator):
    inst, sol = inst_sol_circular_ass_routed_generator(num_carriers=3,
                                                       num_requests_per_carrier=6,
                                                       num_ass_per_carrier=4,
                                                       num_routed_per_carrier=3,
                                                       depot_radius=100,
                                                       requests_radius=120)
    pl.plot_solution_2(inst, sol, show=True)
