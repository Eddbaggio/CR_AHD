import datetime as dt
import json
from typing import List, Sequence, Dict

import numpy as np

import utility_module.io as io
from core_module import instance as it, tour as tr
from utility_module import utils as ut


class CAHDSolution:
    # default, empty solution
    def __init__(self, instance: it.MDPDPTWInstance):
        self.id_: str = instance.id_
        self.meta: Dict[str, int] = instance.meta

        # requests that are not assigned to any carrier
        self.unassigned_requests: List[int] = list(instance.requests)

        # the current REQUEST-to-carrier (not vertex-to-carrier) allocation, initialized with nan for all requests
        self.request_to_carrier_assignment: List[int] = [None for _ in range(instance.num_requests)]

        self.tours: List[tr.Tour] = []
        self.tours_pendulum: List[
            tr.Tour] = []  # TODO urgently need to check whether this is treated correctly in all methods
        self.carriers: List[AHDSolution] = [AHDSolution(c) for c in range(instance.num_carriers)]

        # solver configuration and other meta data
        self.solver_config = dict()
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

    def __repr__(self):
        return f'CAHDSolution for {self.id_}'

    def sum_travel_distance(self):
        return sum(c.sum_travel_distance() for c in self.carriers)

    def sum_travel_duration(self):
        return sum((c.sum_travel_duration() for c in self.carriers), dt.timedelta(0))

    def sum_load(self):
        return sum(c.sum_load() for c in self.carriers)

    def sum_revenue(self):
        return sum(c.sum_revenue() for c in self.carriers)

    def objective(self):
        return sum(c.objective() for c in self.carriers)

    def sum_profit(self):
        return sum(c.sum_profit() for c in self.carriers)

    def num_carriers(self):
        return len(self.carriers)

    def num_tours(self):
        return len(self.tours)

    def num_pendulum_tours(self):
        return len(self.tours_pendulum)

    def num_routing_stops(self):
        return sum(c.num_routing_stops() for c in self.carriers)

    def avg_acceptance_rate(self):
        # average over all carriers
        return sum([c.acceptance_rate for c in self.carriers]) / self.num_carriers()

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carriers}

    def summary(self):
        summary = {**self.meta, }
        summary.update(self.solver_config)
        summary.update({
            # 'num_carriers': self.num_carriers(),
            'objective': self.objective(),
            'sum_profit': self.sum_profit(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'num_tours': self.num_tours(),
            'num_pendulum_tours': self.num_pendulum_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'acceptance_rate': self.avg_acceptance_rate(),
            **self.timings,
            'carrier_summaries': {c.id_: c.summary() for c in self.carriers}
        })

        return summary

    def write_to_json(self):
        path = io.output_dir.joinpath(f'{self.num_carriers()}carriers',
                                      self.id_ + '_' + self.solver_config['solution_algorithm'])
        path = io.unique_path(path.parent, path.stem + '_#{:03d}' + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode='w') as f:
            json.dump({'summary': self.summary(), 'solution': self.as_dict()}, f, indent=4, cls=io.MyJSONEncoder)
        pass

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
            carrier: AHDSolution = self.carriers[self.request_to_carrier_assignment[request]]
            tour = self.tour_of_request(request)
            tour.pop_and_update(instance, [tour.vertex_pos[v] for v in instance.pickup_delivery_pair(request)])
            tour.requests.remove(request)

            # retract the request from the carrier
            carrier.assigned_requests.remove(request)
            if request in carrier.accepted_requests:
                carrier.accepted_requests.remove(request)
            else:
                carrier.accepted_infeasible_requests.remove(request)
            carrier.routed_requests.remove(request)
            self.request_to_carrier_assignment[request] = np.nan
            self.unassigned_requests.append(request)

    def clear_carrier_routes(self, carrier_ids):
        """
        delete all existing routes of the given carrier and move all accepted requests to the list of unrouted requests
        :param carrier_ids:
        """
        if carrier_ids is None:
            carrier_ids = [carrier.id_ for carrier in self.carriers]

        for carrier_id in carrier_ids:
            carrier = self.carriers[carrier_id]
            carrier.unrouted_requests = carrier.accepted_requests[:] + carrier.accepted_infeasible_requests[:]
            carrier.routed_requests.clear()
            self.tours = [tour for tour in self.tours if tour not in carrier.tours + carrier.tours_pendulum]
            for index, tour in enumerate(self.tours):
                tour.id_ = index
            carrier.tours.clear()
            carrier.tours_pendulum.clear()

    def drop_empty_tours_and_adjust_ids(self):
        """
        drops all tours that only contain a depot and do not visit any customer location. removes them from
        self as well as all carriers in self.carriers
        """
        self.tours = [tour for tour in self.tours if tour.requests]
        for index, tour in enumerate(self.tours):
            tour.id_ = index
        for carrier in self.carriers:
            carrier.drop_empty_tours()

    def get_free_pendulum_tour_id(self):
        if None in self.tours_pendulum:
            tour_id = self.tours_pendulum.index(None)
        else:
            tour_id = len(self.tours_pendulum)
        return f'p{tour_id}'

    def get_free_tour_id(self):
        if None in self.tours:
            tour_id = self.tours.index(None)
        else:
            tour_id = len(self.tours)
        return tour_id

    def tour_of_request(self, request: int) -> tr.Tour:
        for tour in self.tours + self.tours_pendulum:
            if request in tour.requests:
                return tour
        return None


class AHDSolution:
    def __init__(self, carrier_index):
        self.id_ = carrier_index
        # self.depots = [carrier_index]  # maybe it's better to store the depots in this class rather than in CAHD?!
        self.assigned_requests: List = []
        self.accepted_requests: List = []
        self.accepted_infeasible_requests: List = []
        self.rejected_requests: List = []
        self.unrouted_requests: List = []
        self.routed_requests: List = []
        self.acceptance_rate: float = 0
        self.tours: List[tr.Tour] = []
        self.tours_pendulum: List[tr.Tour] = []  # 1 for each accepted infeasible request

    def __str__(self):
        s = f'---// Carrier ID: {self.id_} //---' \
            f'\tObjective={round(self.objective(), 4)}, ' \
            f'Acceptance Rate={round(self.acceptance_rate, 2)}, ' \
            f'Assigned={self.assigned_requests}, ' \
            f'Accepted={self.accepted_requests}, ' \
            f'Accepted_Infeasible={self.accepted_infeasible_requests}' \
            f'Unrouted={self.unrouted_requests}, ' \
            f'Routed={self.routed_requests}'
        s += '\n'
        for tour in self.tours:
            s += str(tour)
            s += '\n'
        return s

    def __repr__(self):
        return f'Carrier (AHDSolution) {self.id_}'

    def num_routing_stops(self):
        regular = sum(t.num_routing_stops for t in self.tours)
        pendulum = sum(t.num_routing_stops for t in self.tours_pendulum)
        return regular + pendulum

    def sum_travel_distance(self):
        regular = sum(t.sum_travel_distance for t in self.tours)
        pendulum = sum(t.sum_travel_distance for t in self.tours_pendulum)
        return regular + pendulum

    def sum_travel_duration(self):
        regular = sum((t.sum_travel_duration for t in self.tours), dt.timedelta(0))
        pendulum = sum((t.sum_travel_duration for t in self.tours_pendulum), dt.timedelta(0))
        return regular + pendulum

    def sum_load(self):
        regular = sum(t.sum_load for t in self.tours)
        pendulum = sum(t.sum_load for t in self.tours_pendulum)
        return regular + pendulum

    def sum_revenue(self):
        regular = sum(t.sum_revenue for t in self.tours)
        pendulum = sum(t.sum_revenue for t in self.tours_pendulum)
        return regular + pendulum

    def sum_profit(self):
        regular = sum(t.sum_profit for t in self.tours)
        pendulum = sum(
            t.sum_revenue - t.sum_travel_distance * ut.PENDULUM_PENALTY_DISTANCE_SCALING for t in self.tours_pendulum)
        return regular + pendulum

    def objective(self):
        return self.sum_profit()

    def as_dict(self):
        return {
            # 'id_': self.id_,
            'tours': {
                tour.id_: tour.as_dict() for tour in self.tours
            },
            'pendulum_tours': {
                tour.id_: tour.as_dict() for tour in self.tours_pendulum
            }
        }

    def summary(self) -> dict:
        return {
            'carrier_id': self.id_,
            'num_tours': len(self.tours),
            'num_routing_stops': self.num_routing_stops(),
            'sum_profit': self.objective(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'acceptance_rate': self.acceptance_rate,
            'num_pendulum_tours': len(self.tours_pendulum),
            'tour_summaries': {t.id_: t.summary() for t in self.tours},
            'tours_pendulum_summaries': {t.id_: t.summary() for t in self.tours_pendulum}
        }

    def drop_empty_tours(self):
        """
        drops all tours that only contain a depot and do not visit any customer location. removes them from
        self
        """
        self.tours = [tour for tour in self.tours if tour.requests]
        pass