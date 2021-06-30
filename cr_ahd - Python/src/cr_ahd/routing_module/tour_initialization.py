import random
from abc import ABC, abstractmethod
from typing import Sequence, List

import numpy as np

from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut


# =====================================================================================================================
# single-criterion-based
# =====================================================================================================================

class TourInitializationBehavior(ABC):
    """
    Visitor Interface to apply a tour initialization heuristic to either an instance (i.e. each of its carriers)
    or a single specific carrier.
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


# =====================================================================================================================
# Graph-based
# =====================================================================================================================

class MaxCliqueTourInitializationBehavior(ABC):
    """
    based on
    [1] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
    delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
    https://doi.org/10.1016/j.ejor.2005.05.012
    """

    def execute(self, instance: it.PDPInstance, solution: slt.CAHDSolution):
        for carrier in range(instance.num_carriers):
            self._initialize_carrier(instance, solution, carrier)
        pass

    def _initialize_carrier(self, instance: it.PDPInstance, solution: slt.CAHDSolution, carrier: int):
        carrier_ = solution.carriers[carrier]
        assert len(solution.carrier_depots[carrier]) == 1, f'graph based initialization only available for single-depot'
        assert carrier_.unrouted_requests

        # find candidates for initial pendulum tours

        # create the graph G_N
        g_nodes = list(range(len(carrier_.unrouted_requests)))  # each request is a node
        g_arcs = self.g_arcs(instance, solution, carrier)

        max_clique = self.max_clique(g_nodes, g_arcs)  # returns indices corresponding to the list of unrouted!

        assert len(max_clique) <= instance.carriers_max_num_tours

        # create the pendulum tours, popping the seeds of the list of unrouted requires reverse traversal
        for i in sorted(max_clique, reverse=True):
            seed = carrier_.unrouted_requests[i]
            tour = tr.Tour(carrier_.num_tours(), instance, solution, solution.carrier_depots[carrier][0])
            tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(seed))
            carrier_.tours.append(tour)
            carrier_.unrouted_requests.pop(i)

        pass

    def g_arcs(self, instance, solution, carrier):
        carrier_ = solution.carriers[carrier]
        num_requests = len(carrier_.unrouted_requests)

        g_arcs = [[0] * num_requests for _ in range(num_requests)]  # n*n matrix of zeros
        # g_arcs = np.zeros((instance.num_carriers, instance.num_carriers))  # twice as fast

        # create arcs for request pairs if and only if no vehicle can serve i_request and j_request in a single tour
        for i, i_request in enumerate(carrier_.unrouted_requests[:-1]):
            i_pickup, i_delivery = instance.pickup_delivery_pair(i_request)

            for j, j_request in enumerate(carrier_.unrouted_requests[1:], start=1):
                j_pickup, j_delivery = instance.pickup_delivery_pair(j_request)

                tour_ = tr.Tour('tmp', instance, solution, solution.carrier_depots[carrier][0])

                # make sure that the first request is feasible alone
                assert tour_.insertion_feasibility_check(instance, solution, [1, 2], [i_pickup, i_delivery])
                tour_.insert_and_update(instance, solution, [1, 2], [i_pickup, i_delivery])

                # try all insertion positions to whether all are infeasible
                feasible = False
                for j_pickup_pos in range(1, 4):
                    for j_delivery_pos in range(j_pickup_pos + 1, 5):

                        # connect the nodes if insertion is infeasible
                        if tour_.insertion_feasibility_check(instance,
                                                             solution,
                                                             [j_pickup_pos, j_delivery_pos],
                                                             [j_pickup, j_delivery]):
                            feasible = True
                            break

                    if feasible:
                        break

                # only if no feasible insertion for j_request was found, an arc is created
                if not feasible:
                    g_arcs[i][j] = 1
                    g_arcs[j][i] = 1  # create both triangles of the matrix

        return g_arcs

    def _request_evaluation(self, pickup_idx: int, delivery_idx: int, **kwargs):
        pass

    def max_clique(self, g_nodes: List[int], g_arcs: List[List[int]]):
        """
        will modify g_nodes and g_arcs in place!
        :param g_nodes:
        :param g_arcs:
        :return:
        """
        max_clique = None
        max_max_degree = 0

        # [1] choose each node as the initial node once

        """
        for i_idx, i_node in enumerate(g_nodes):
        """

        marked = [False] * len(g_nodes)

        """
            marked[i] = True
            # [2] remove nodes that have no edge linking edge to initially selected i
            for j in g_nodes:

                if i == j:
                    continue

                if g_arcs_copy[i][j] == 0:
                    clique.pop(j)
                    marked.pop(j)
                    for k in g_nodes:
                        g_arcs_copy[k][j] = [0] * len(g_arcs_copy)
            """

        while not all(marked):

            # [3] select & mark an arbitrary node with max number of incident edges
            unmarked_max_degree = max([sum(w) for v, w in enumerate(g_arcs) if marked[v] is False])

            """
            # stop if current clique has lower degree than the max_clique
            if unmarked_max_degree <= max_max_degree:
                break
            """

            unmarked_max_degree_nodes = [i for i in g_nodes if
                                         sum(g_arcs[i]) == unmarked_max_degree and marked[i] is False]
            i = random.choice(unmarked_max_degree_nodes)  # TODO why do this randomly?! no need
            marked[i] = True

            # [4] remove nodes that have no edge linking to i
            for j in g_nodes.copy():

                if i == j:
                    continue

                if g_arcs[i][j] == 0:

                    # remove arcs of j_node
                    for k in g_nodes:
                        g_arcs[k][j] = 0
                        g_arcs[j][k] = 0

                    # remove j from graph nodes
                    g_nodes.remove(j)

                    # mark j
                    marked[j] = True

                    """
                    max_clique = clique
            max_max_degree = unmarked_max_degree
            """

        return g_nodes


if __name__ == '__main__':
    MaxCliqueTourInitializationBehavior().max_clique(
        list(range(6)),
        [
            [0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0]
        ]
    )
