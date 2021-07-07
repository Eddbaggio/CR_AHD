import json
from typing import List, Sequence

import numpy as np

from src.cr_ahd.core_module import instance as it, tour as tr
from src.cr_ahd.utility_module import utils as ut
import datetime as dt


class CAHDSolution:
    # default, empty solution
    def __init__(self, instance: it.PDPInstance):
        self.id_ = instance.id_
        self.meta = instance.meta

        # requests that are not assigned to any carrier
        self.unassigned_requests = list(instance.requests)

        # the current REQUEST-to-carrier (not vertex-to-carrier) allocation, initialized with nan for all requests
        self.request_to_carrier_assignment = np.full(instance.num_requests, np.nan)

        # basically no apriori time windows for all VERTICES
        self.tw_open = np.full(instance.num_depots + 2 * instance.num_requests, ut.START_TIME).tolist()
        self.tw_close = np.full(instance.num_depots + 2 * instance.num_requests, ut.END_TIME).tolist()

        self.carriers = [AHDSolution(c) for c in range(instance.num_carriers)]

        # one depot per carrier, can be adjusted externally for multi-depot, single-carrier problems
        self.carrier_depots = [[depot] for depot in range(instance.num_depots)]

        # stuff that is filled during the solving
        self.solution_algorithm = None
        self.auction_mechanism = None

    def __str__(self):
        s = f'Solution {self.id_}\nProfit={round(self.sum_profit(), 2)}'
        s += '\n'
        for c in self.carriers:
            s += str(c)
            s += '\n'
        return s

    @property
    def unrouted_requests(self):
        return set().union(*[c.unrouted_requests for c in self.carriers])

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

    def sum_profit(self):
        return sum(c.sum_profit() for c in self.carriers)

    def num_carriers(self):
        return len(self.carriers)

    def num_tours(self):
        return sum(c.num_tours() for c in self.carriers)

    def num_routing_stops(self):
        return sum(c.num_routing_stops() for c in self.carriers)

    def acceptance_rate(self):
        return sum([c.acceptance_rate for c in self.carriers])/self.num_carriers()

    def assign_requests_to_carriers(self, requests: Sequence[int], carriers: Sequence[int]):
        for r, c in zip(requests, carriers):
            self.request_to_carrier_assignment[r] = c
            self.unassigned_requests.remove(r)
            self.carriers[c].assigned_requests.append(r)
            self.carriers[c].unrouted_requests.append(r)

    def clear_carrier_routes(self):
        """delete all existing routes and move all accepted requests to the list of unrouted requests"""
        for carrier_ in self.carriers:
            carrier_.clear_routes()

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carriers}

    def summary(self):
        return {
            'id_': self.id_,
            'solution_algorithm': self.solution_algorithm,
            'num_carriers': self.num_carriers(),
            'carrier_depots': self.carrier_depots,
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            # 'sum_wait_duration': self.sum_wait_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'acceptance_rate': self.acceptance_rate(),
            'carrier_summaries': {c.id_: c.summary() for c in self.carriers}
        }

    def write_to_json(self):
        path = ut.output_dir_GH.joinpath(f'{self.num_carriers()}carriers',
                                         self.id_ + '_' + self.solution_algorithm)
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
        self.tours: List[tr.Tour] = []

    def __str__(self):
        s = f'---// Carrier ID: {self.id_} //---' \
            f'\tProfit={round(self.sum_profit(), 4)}, Acceptance Rate={round(self.acceptance_rate, 2)}, ' \
            f'Assigned={self.assigned_requests}, Accepted={self.accepted_requests},Unrouted={self.unrouted_requests}'
        s += '\n'
        for tour_ in self.tours:
            s += str(tour_)
            s += '\n'
        return s

    def num_routing_stops(self):
        return sum(t.num_routing_stops for t in self.tours)

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

    def sum_profit(self):
        return sum(t.sum_profit for t in self.tours)

    def num_tours(self):
        return len(self.tours)

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
            'sum_profit': self.sum_profit(),
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
