import datetime as dt
import json
from typing import List, Sequence

import numpy as np

from src.cr_ahd.core_module import instance as it, tour as tr
from src.cr_ahd.utility_module import utils as ut


class CAHDSolution:
    # default, empty solution
    def __init__(self, instance: it.MDPDPTWInstance):
        self.id_ = instance.id_
        self.meta = instance.meta

        # requests that are not assigned to any carrier
        self.unassigned_requests = list(instance.requests)

        # the current REQUEST-to-carrier (not vertex-to-carrier) allocation, initialized with nan for all requests
        self.request_to_carrier_assignment: List[int] = [None for _ in range(instance.num_requests)]
        # store a lookup-table to get a request's tour index
        self.request_to_tour_assignment: List[int] = [None for _ in range(instance.num_requests)]
        # store a lookup-table to get vertex's position inside its tour
        # self.vertex_position_in_tour: List[int] = [None for _ in range(instance.num_depots + instance.num_requests * 2)]

        self.tours: List[tr.Tour] = []
        self.carriers = [AHDSolution(c) for c in range(instance.num_carriers)]

        # solver configuration and other meta data
        self.solver_config = {config: None for config in ut.solver_config}
        self.timings = dict()
        # TODO find a better way to add available ls neighborhoods automatically
        self.local_search_move_counter = dict(PDPMove=0,
                                              PDPTwoOpt=0,
                                              PDPRelocate=0,
                                              PDPRelocate2=0,
                                              PDPLargeInterTourNeighborhood=0)

    def __str__(self):
        s = f'Solution {self.id_}\nObjective={round(self.objective(), 2)}'
        s += '\n'
        for c in self.carriers:
            s += str(c)
            s += '\n'
        return s

    def sum_travel_distance(self):
        return sum(c.sum_travel_distance() for c in self.carriers)

    def sum_travel_duration(self):
        return sum((c.sum_travel_duration() for c in self.carriers), dt.timedelta(0))

    # def sum_wait_duration(self):
    #     return sum((c.sum_wait_duration() for c in self.carriers), dt.timedelta(0))

    def sum_load(self):
        return sum(c.sum_load() for c in self.carriers)

    def sum_revenue(self):
        return sum(c.sum_revenue() for c in self.carriers)

    def objective(self):
        return sum(c.objective() for c in self.carriers)

    def tmp_sum_profit(self):
        return sum(c.objective() for c in self.carriers)

    def num_carriers(self):
        return len(self.carriers)

    def num_tours(self):
        return len(self.tours)

    def num_routing_stops(self):
        return sum(c.num_routing_stops() for c in self.carriers)

    def avg_acceptance_rate(self):
        # average over all carriers
        return sum([c.acceptance_rate for c in self.carriers]) / self.num_carriers()

    def assign_requests_to_carriers(self, requests: Sequence[int], carriers: Sequence[int]):
        for r, c in zip(requests, carriers):
            self.request_to_carrier_assignment[r] = c
            self.unassigned_requests.remove(r)
            self.carriers[c].assigned_requests.append(r)
            self.carriers[c].unrouted_requests.append(r)

    def free_requests_from_carriers(self, instance: it.MDPDPTWInstance, requests: Sequence[int]):
        """
        removes the given requests from their route and sets them to be unassigned and not accepted (not_accepted !=
        rejected)

        :param instance:
        :param requests:
        :return:
        """
        for request in requests:
            carrier_: AHDSolution = self.carriers[self.request_to_carrier_assignment[request]]
            tour_ = self.tours[self.request_to_tour_assignment[request]]
            for vertex in instance.pickup_delivery_pair(request):
                pos = tour_._vertex_pos[vertex]
                tour_.pop_and_update(instance, self, [pos])

            self.request_to_tour_assignment[request] = None

            # retract the request from the carrier
            carrier_.assigned_requests.remove(request)
            carrier_.accepted_requests.remove(request)
            carrier_.routed_requests.remove(request)
            self.request_to_carrier_assignment[request] = np.nan
            self.unassigned_requests.append(request)

    def unroute_request(self, instance: it.MDPDPTWInstance, requests: List[int]):
        """
        remove the selected requests from their current tour and set them as unassigned without retracting them from
        the carrier. Thus, the given requests will be unrouted

        :param instance:
        :param requests:
        :return:
        """

    def clear_carrier_routes(self):
        """delete all existing routes and move all accepted requests to the list of unrouted requests"""
        for carrier_ in self.carriers:
            carrier_.clear_routes()

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carriers}

    def summary(self):
        summary = {
            'id_': self.id_,
            'num_carriers': self.num_carriers(),
            'carrier_depots': self.carrier_depots,
            'objective': self.objective(),
            'sum_profit': self.tmp_sum_profit(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            # 'sum_wait_duration': self.sum_wait_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'acceptance_rate': self.avg_acceptance_rate(),
            **self.timings,
            'carrier_summaries': {c.id_: c.summary() for c in self.carriers}
        }
        summary.update(self.solver_config)
        return summary

    def write_to_json(self):
        path = ut.output_dir_GH.joinpath(f'{self.num_carriers()}carriers',
                                         self.id_ + '_' + self.solver_config['solution_algorithm'])
        path = ut.unique_path(path.parent, path.stem + '_#{:03d}' + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode='w') as f:
            json.dump({'summary': self.summary(), 'solution': self.as_dict()}, f, indent=4, cls=ut.MyJSONEncoder)
        pass


class AHDSolution:
    def __init__(self, carrier_index):
        self.id_ = carrier_index
        # self.depots = [carrier_index]  # maybe it's better to store the depots in this class rather than in CAHD?!
        self.assigned_requests: List = []
        self.accepted_requests: List = []
        self.rejected_requests: List = []
        self.unrouted_requests: List = []
        self.routed_requests: List = []
        self.acceptance_rate: float = 0
        self.tour_ids: List[int] = []   # list of tour IDs of the tours that belong to this solution/carrier

    def __str__(self):
        s = f'---// Carrier ID: {self.id_} //---' \
            f'\tObjective={round(self.objective(), 4)}, ' \
            f'Acceptance Rate={round(self.acceptance_rate, 2)}, ' \
            f'Assigned={self.assigned_requests}, ' \
            f'Accepted={self.accepted_requests}, ' \
            f'Unrouted={self.unrouted_requests}, ' \
            f'Routed={self.routed_requests}'
        s += '\n'
        for tour_ in self.tours():
            s += str(tour_)
            s += '\n'
        return s

    def tours(self, solution: CAHDSolution):
        return (solution.tours[x] for x in self.tour_ids)

    def num_routing_stops(self, solution: CAHDSolution):
        return sum(t.num_routing_stops for t in self.tours(solution))

    def sum_travel_distance(self):
        return sum(t.sum_travel_distance for t in self.tours)

    def sum_travel_duration(self):
        return sum((t.sum_travel_duration for t in self.tours), dt.timedelta(0))

    # def sum_wait_duration(self):
    #     return sum((t.sum_wait_duration for t in self.tours), dt.timedelta(0))

    def sum_load(self):
        return sum(t.sum_load for t in self.tours)

    def sum_revenue(self):
        return sum(t.sum_revenue for t in self.tours)

    def objective(self):
        return sum(t.sum_profit for t in self.tours)

    def num_tours(self):
        return len(self.tour_ids)

    def as_dict(self):
        return {
            # 'id_': self.id_,
            'tours': {
                tour.id_: tour.as_dict() for tour in self.tours
            },
        }

    def summary(self) -> dict:
        return {
            # 'id_': self.id_,
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'sum_profit': self.objective(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            # 'sum_wait_duration': self.sum_wait_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'acceptance_rate': self.acceptance_rate,
            'tour_summaries': {t.id_: t.summary() for t in self.tours}
        }

    def clear_routes(self):
        self.unrouted_requests = self.accepted_requests[:]
        self.routed_requests.clear()
        self.tours.clear()
