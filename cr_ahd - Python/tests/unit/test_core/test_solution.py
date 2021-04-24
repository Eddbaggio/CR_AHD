from src.cr_ahd.core_module import instance as it, solution as slt


class TestGlobalSolution:
    def test_unrouted_requests(self, sol_gh_0: slt.GlobalSolution):
        assert sol_gh_0.unrouted_requests == list(range(15))

    def test_assign_requests_to_carriers(self, sol_gh_0: slt.GlobalSolution):
        solution = sol_gh_0
        req = [1, 5, 6, 14]
        carrier = [2, 0, 1, 1]
        solution.assign_requests_to_carriers(req, carrier)
        for c, r in zip(req, carrier):
            assert solution.request_to_carrier_assignment[r] == c
            assert r in solution.carrier_solutions[c].unrouted_requests
        assert solution.unassigned_requests == 15 - len(req)

class TestPDPSolution:
    pass