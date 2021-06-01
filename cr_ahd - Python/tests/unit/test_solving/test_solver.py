import src.cr_ahd.solver as sl
from src.cr_ahd.core_module import instance as it, solution as slt


class TestDynamicSolver:
    def test_assign_n_requests(self, inst_gh_0: it.PDPInstance, sol_gh_0: slt.CAHDSolution):
        sl.assign_n_requests(inst_gh_0, sol_gh_0, 3)
        assert sol_gh_0.carriers[0].unrouted_requests == [0, 1, 2]
        assert sol_gh_0.carriers[1].unrouted_requests == [5, 6, 7]
        assert sol_gh_0.carriers[2].unrouted_requests == [10, 11, 12]
        assert sol_gh_0.unassigned_requests == [3, 4, 8, 9, 13, 14]