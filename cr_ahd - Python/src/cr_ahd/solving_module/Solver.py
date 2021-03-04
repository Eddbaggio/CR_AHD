from abc import ABC, abstractmethod
from src.cr_ahd.core_module import instance as it
from src.cr_ahd.utility_module.utils import get_carrier_by_id, InsertionError
from src.cr_ahd.solving_module.initializing_visitor import EarliestDueDate
from src.cr_ahd.solving_module.routing_visitor import SequentialCheapestInsertion, StaticI1Insertion
from src.cr_ahd.solving_module.local_search_visitor import TwoOpt


class Solver(ABC):
    @abstractmethod
    def solve(self, instance: it.Instance):
        pass


class StaticSequentialInsertionSolver(Solver):
    def solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        self.build_routes(instance)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_request(request)

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def build_routes(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().finalize_instance(instance)
        pass


class StaticI1InsertionSolver(Solver):
    def solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        self.build_routes(instance)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_request(request)

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def build_routes(self, instance):
        StaticI1Insertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().finalize_instance(instance)
        pass


class DynamicSequentialInsertionSolver(Solver):
    def solve(self, instance: it.Instance):
        while instance.unrouted_requests:
            self.assign_n_requests(instance, 5)
            self.build_routes(instance)
            instance._solved = False
        instance.solved = True
        self.finalize_with_local_search(instance)
        pass

    def assign_n_requests(self, instance, n):
        for request in instance.unassigned_requests()[:n]:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_request(request)
        pass

    def build_routes(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().finalize_instance(instance)
        pass
