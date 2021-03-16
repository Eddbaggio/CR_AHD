import abc

import src.cr_ahd.auction_module.request_selection as rs
import src.cr_ahd.auction_module.bundle_generation as bg
import src.cr_ahd.auction_module.bidding as bd
import src.cr_ahd.auction_module.winner_determination as wd

from src.cr_ahd.core_module import instance as it
from src.cr_ahd.solving_module.tour_construction import SequentialCheapestInsertion, I1Insertion
from src.cr_ahd.solving_module.tour_improvement import TwoOpt
from src.cr_ahd.solving_module.tour_initialization import EarliestDueDate
from src.cr_ahd.utility_module.utils import get_carrier_by_id, opts


# crude Template method implementation


class Solver(abc.ABC):
    def execute(self, instance):
        """
        apply the concrete solution algorithm
        :param instance:
        :return: None
        """
        self._solve(instance)
        instance.solution_algorithm = self
        pass

    @abc.abstractmethod
    def _solve(self, instance: it.Instance):
        pass


class StaticSequentialInsertion(Solver):
    """
    Initializes pendulum tours first! Then builds the routes. The Dynamic Version does NOT initialize tours
    """
    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        self.build_routes(instance)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def build_routes(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class StaticI1Insertion(Solver):
    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        self.initialize_pendulum_tours(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)
        self.finalize_with_local_search(instance)
        pass

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def initialize_pendulum_tours(self, instance):
        EarliestDueDate().initialize_instance(instance)  # not flexible!
        pass

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class StaticI1InsertionWithAuction(Solver):
    def _solve(self, instance: it.Instance):
        self.assign_all_requests(instance)
        if instance.num_carriers > 1:
            self.run_auction(instance)
        carrier = True
        while carrier:
            carrier = self.build_routes(instance)
            if carrier:
                self.initialize_another_tour(carrier)
        self.finalize_with_local_search(instance)

    def assign_all_requests(self, instance):
        for request in instance.unrouted_requests:
            carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
            carrier.assign_requests([request])

    def run_auction(self, instance):
        submitted_requests = rs.FiftyPercentHighestMarginalCost().execute(instance)
        bundle_set = bg.RandomBundles(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1MarginalCostBidding().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)


# =====================================================================================================================
# =====================================================================================================================


def assign_n_requests(instance, n):
    for request in instance.unassigned_requests()[:n]:
        carrier = get_carrier_by_id(instance.carriers, request.carrier_assignment)
        carrier.assign_requests([request])


class DynamicSequentialInsertion(Solver):
    """
    Does NOT initialize pendulum tours!
    """
    def _solve(self, instance: it.Instance):
        while instance.unrouted_requests:
            assign_n_requests(instance, opts['dynamic_cycle_time'])
            self.build_routes(instance)
            # instance._solved = False
            # self.finalize_with_local_search(instance)
            # instance._finalized = False
        # instance.solved = True
        self.finalize_with_local_search(instance)
        pass

    def build_routes(self, instance):
        SequentialCheapestInsertion().solve_instance(instance)
        pass

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class DynamicI1Insertion(Solver):
    def _solve(self, instance: it.Instance):
        while instance.unrouted_requests:
            assign_n_requests(instance, opts['dynamic_cycle_time'])
            carrier = True
            while carrier:
                carrier = self.build_routes(instance)
                if carrier:
                    self.initialize_another_tour(carrier)
            # instance._solved = False
            # self.finalize_with_local_search(instance)
            # instance._finalized = False
        # instance.solved = True
        self.finalize_with_local_search(instance)
        pass

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
        pass


class DynamicI1InsertionWithAuction(Solver):
    def _solve(self, instance: it.Instance):
        while instance.unrouted_requests:
            assign_n_requests(instance, opts['dynamic_cycle_time'])
            if instance.num_carriers > 1:
                self.run_auction(instance)
            carrier = True
            while carrier:
                carrier = self.build_routes(instance)
                if carrier:
                    self.initialize_another_tour(carrier)
        self.finalize_with_local_search(instance)

    def run_auction(self, instance):
        submitted_requests = rs.FiftyPercentHighestMarginalCost().execute(instance)
        bundle_set = bg.RandomBundles(instance.distance_matrix).execute(submitted_requests)
        bids = bd.I1MarginalCostBidding().execute(bundle_set, instance.carriers)
        wd.LowestBid().execute(bids)

    def initialize_another_tour(self, carrier):
        EarliestDueDate().initialize_carrier(carrier)

    def build_routes(self, instance):
        carrier = I1Insertion().solve_instance(instance)
        return carrier

    def finalize_with_local_search(self, instance):
        TwoOpt().improve_instance(instance)
