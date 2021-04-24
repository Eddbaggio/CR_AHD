import pytest

import src.cr_ahd.utility_module.plotting as pl
from src.cr_ahd.core_module import solution as slt

SKIP_TEST_PLOTTING = False


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_inst_gh_0(inst_gh_0):
    solution = slt.GlobalSolution(inst_gh_0)
    pl.plot_solution_2(inst_gh_0, solution, show=True)
    pass


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_circular_inst_gen(instance_generator_circular):
    inst = instance_generator_circular()
    solution = slt.GlobalSolution(inst)
    pl.plot_solution_2(inst, solution, show=True)
    pass


@pytest.mark.skipif(SKIP_TEST_PLOTTING, reason='Deactivated via SKIP_TEST_PLOTTING')
def test_plot_inst_and_sol_gh_0_9_ass_6_routed(inst_and_sol_gh_0_9_ass_6_routed):
    inst, solution = inst_and_sol_gh_0_9_ass_6_routed
    pl.plot_solution_2(inst, solution, show=True)
    pass
