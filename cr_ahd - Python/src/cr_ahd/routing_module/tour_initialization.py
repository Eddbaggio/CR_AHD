from abc import ABC, abstractmethod

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut


class TourInitializationBehavior(ABC):
    """
    Visitor Interface to apply a tour initialization heuristic to either an instance (i.e. each of its carriers)
    or a single specific carrier. Contrary to the routing visitor, this one will only allocate a single seed request
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        for carrier in range(instance.num_carriers):
            self._initialize_carrier(instance, solution, carrier)
        pass

    def _initialize_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        carrier_ = solution.carriers[carrier]
        assert carrier_.unrouted_requests

        # create (potentially multiple) initial pendulum tour(s)
        num_pendulum_tours = instance.carriers_max_num_tours
        # TODO the num_pendulum_tours should probably not be fixed, but be dependent on the size of the max clique
        for pendulum_tour in range(num_pendulum_tours):

            best_request = None
            best_depot = None
            best_evaluation = -float('inf')

            for request in carrier_.unrouted_requests:

                depot_and_evaluations = []
                for depot in solution.carrier_depots[carrier]:
                    evaluation = self._request_evaluation(*instance.pickup_delivery_pair(request),
                                                          **{'x_depot': instance.x_coords[depot],
                                                             'y_depot': instance.y_coords[depot],
                                                             'x_coords': instance.x_coords,
                                                             'y_coords': instance.y_coords,
                                                             })
                    depot_and_evaluations.append((depot, evaluation))

                depot, evaluation = min(depot_and_evaluations, key=lambda x: x[1])

                # update the best known seed
                if evaluation > best_evaluation:
                    best_request = request
                    best_depot = depot
                    best_evaluation = evaluation

            # create the pendulum tour
            tour = tr.Tour(carrier_.num_tours(), instance, solution, best_depot)
            tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(best_request))
            carrier_.tours.append(tour)
            carrier_.unrouted_requests.remove(best_request)
        pass

    @abstractmethod
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs):
        r"""
        :param pickup_idx:
        :param delivery_idx:
        :param kwargs: \**kwargs:
        See below

        :Keyword Arguments:
        * *x_depot*  --
        * *y_depot*  --
        * *x_coords*  --
        * *y_coords*  --
        * *tw_open*  --
        * *tw_close*  --

        :return:
        """
        pass


class EarliestDueDate(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        return - kwargs['tw_close'][delivery_idx].total_seconds


class FurthestDistance(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        x_midpoint, y_midpoint = ut.midpoint_(kwargs['x_coords'][pickup_idx], kwargs['x_coords'][delivery_idx],
                                              kwargs['y_coords'][pickup_idx], kwargs['y_coords'][delivery_idx])
        return ut.euclidean_distance(kwargs['x_depot'], kwargs['y_depot'], x_midpoint, y_midpoint)


class ClosestDistance(TourInitializationBehavior):
    def _request_evaluation(self,
                            pickup_idx: int,
                            delivery_idx: int,
                            **kwargs
                            ):
        x_midpoint, y_midpoint = ut.midpoint_(kwargs['x_coords'][pickup_idx], kwargs['x_coords'][delivery_idx],
                                              kwargs['y_coords'][pickup_idx], kwargs['y_coords'][delivery_idx])
        return - ut.euclidean_distance(kwargs['x_depot'], kwargs['y_depot'], x_midpoint, y_midpoint)
