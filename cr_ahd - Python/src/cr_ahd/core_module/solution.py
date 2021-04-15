import json
from pathlib import Path
from typing import List

import numpy as np
import datetime as dt

from src.cr_ahd.core_module import instance as it, tour as tr
from src.cr_ahd.utility_module import utils as ut


class GlobalSolution:
    # default, empty solution
    def __init__(self, instance: it.PDPInstance):
        self.id_ = instance.id_
        self.meta = instance.meta
        # REQUESTS that are not assigned to any carrier (went with a list rather than a set because when re-assigning
        # requests, i need index-based access to the elements
        self.unassigned_requests = list(instance.requests)
        # the current REQUEST-to-carrier (not vertex-to-carrier) allocation, initialized with nan for all requests
        self.request_to_carrier_assignment = np.full(instance.num_requests, np.nan)
        # TODO unassigned_requests and request_to_carrier_assignment contain redundant data.
        #  unassigned_requests == np.where(np.isnan(request_to_carrier_assignment))
        # basically no apriori time windows for all VERTICES
        self.tw_open = np.full(instance.num_carriers + 2 * instance.num_requests, ut.START_TIME)
        self.tw_close = np.full(instance.num_carriers + 2 * instance.num_requests, ut.END_TIME)

        self.carrier_solutions = [PDPSolution(c) for c in range(instance.num_carriers)]

        self.solution_algorithm = None
        self.auction_mechanism = None

    @property
    def unrouted_requests(self):
        return set().union(*[c.unrouted_requests for c in self.carrier_solutions])

    def sum_travel_distance(self):
        return sum(c.sum_travel_distance() for c in self.carrier_solutions)

    def sum_travel_duration(self):
        return np.sum([c.sum_travel_duration() for c in self.carrier_solutions])

    def sum_load(self):
        return sum(c.sum_load() for c in self.carrier_solutions)

    def sum_revenue(self):
        return sum(c.sum_revenue() for c in self.carrier_solutions)

    def sum_profit(self):
        return sum(c.sum_profit() for c in self.carrier_solutions)

    def num_carriers(self):
        return len(self.carrier_solutions)

    def num_tours(self):
        return sum(c.num_tours() for c in self.carrier_solutions)

    def num_routing_stops(self):
        return sum(c.num_routing_stops() for c in self.carrier_solutions)
        pass

    def assign_requests_to_carriers(self, requests: List[int], carriers: List[int]):
        for r, c in zip(requests, carriers):
            self.request_to_carrier_assignment[r] = c
            self.unassigned_requests.remove(r)
            self.carrier_solutions[c].unrouted_requests.append(r)

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carrier_solutions}

    def summary(self):
        return {
            'id_': self.id_,
            'solution_algorithm': self.solution_algorithm,
            'num_carriers': self.num_carriers(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'carrier_summaries': {c.id_: c.summary() for c in self.carrier_solutions}
        }

    def write_to_json(self):
        path = ut.path_output_gansterer.joinpath(f'{self.num_carriers()}carriers',
                                                 self.id_ + '_' + self.solution_algorithm)
        path = ut.unique_path(path.parent, path.stem + '_#{:03d}' + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode='w') as f:
            json.dump({'summary': self.summary(), 'solution': self.as_dict()}, f, indent=4, cls=ut.MyJSONEncoder)
        pass


class PDPSolution:
    def __init__(self, carrier_index):
        self.id_ = carrier_index
        self.unrouted_requests: List = list()  # must be a list (instead of a set) because it must be sorted for bidding (could create a local list-copy though, if i really wanted)
        self.tours: List[tr.Tour] = []

        self.solution_algorithm = None

    def num_routing_stops(self):
        return sum(t.num_routing_stops for t in self.tours)

    def sum_travel_distance(self):
        return sum(t.sum_travel_distance for t in self.tours)

    def sum_travel_duration(self):
        return np.sum([t.sum_travel_duration for t in self.tours])

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

    def summary(self):
        return {
            # 'id_': self.id_,
            'num_tours': self.num_tours(),
            'num_routing_stops': self.num_routing_stops(),
            'sum_profit': self.sum_profit(),
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'tour_summaries': {t.id_: t.summary() for t in self.tours}
        }

    def exchange_vertices(self, instance: it.PDPInstance, solution: GlobalSolution, old_tour: int, new_tour: int,
                          old_positions: List[int],
                          new_positions: List[int]):
        """
        used for the PDPExchange local search operator. Allows to move vertices from one tour to another. In a PDP
        context this will always be pickup-delivery pairs being moved from one tour to another.

        :param instance:
        :param solution:
        :param old_tour:
        :param new_tour:
        :param old_positions:
        :param new_positions:
        :return:
        """
        vertices = self.tours[old_tour].pop_and_update(instance, solution, old_positions)
        self.tours[new_tour].insert_and_update(instance, solution, new_positions, vertices)


def read_solution_and_summary_from_json(path: Path):
    with open(path, mode='r') as f:
        solution, summary = json.load(f).values()
    return solution, summary
