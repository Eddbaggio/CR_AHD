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
        # REQUESTS that are not assigned to any carrier (went with a list rather than a set because when re-assigning
        # requests, i need index-based access to the elements
        self.unassigned_requests = list(instance.requests)
        # the current REQUEST-to-carrier (not vertex-to-carrier) allocation, initialized with -1 for requests
        self.request_to_carrier_assignment = np.full(instance.num_requests, np.nan)
        # basically no apriori time windows for all VERTICES
        self.tw_open = np.full(instance.num_carriers + 2 * instance.num_requests, ut.START_TIME)
        self.tw_close = np.full(instance.num_carriers + 2 * instance.num_requests, ut.END_TIME)
        self.carrier_solutions = [PDPSolution(instance, c) for c in range(instance.num_carriers)]

        self.cost = 0
        self.solution_algorithm = None

    @property
    def unrouted(self):
        return set().union(*[c.unrouted_requests for c in self.carrier_solutions])

    def sum_travel_distance(self):
        return sum(c.sum_travel_distance() for c in self.carrier_solutions)

    def sum_travel_duration(self):
        return np.sum([c.sum_travel_duration() for c in self.carrier_solutions])

    def sum_load(self):
        return sum(c.sum_load() for c in self.carrier_solutions)

    def sum_revenue(self):
        return sum(c.sum_revenue() for c in self.carrier_solutions)

    def num_tours(self):
        return sum(c.num_tours() for c in self.carrier_solutions)

    def assign_requests_to_carriers(self, requests: List[int], carriers: List[int]):
        for r, c in zip(requests, carriers):
            self.request_to_carrier_assignment[r] = c
            self.unassigned_requests.remove(r)
            self.carrier_solutions[c].unrouted_requests.add(r)

    def as_dict(self):
        """The solution as a nested python dictionary"""
        return {carrier.id_: carrier.as_dict() for carrier in self.carrier_solutions}

    def summary(self):
        return {
            # 'id_': self.id_,
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'num_tours': self.num_tours(),
            'carrier_summaries': {c.id_: c.summary() for c in self.carrier_solutions}
        }

    def write_to_json(self):
        path = ut.path_output_gansterer.joinpath(self.id_)
        path = ut.unique_path(path.parent, path.stem + '_#{:03d}' + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode='w') as f:
            json.dump({'summary': self.summary(), 'solution': self.as_dict()}, f, indent=4, cls=ut.MyJSONEncoder)
        pass

    def plot(self):
        pass


class PDPSolution:
    def __init__(self, instance: it.PDPInstance, carrier_index):
        self.id_ = carrier_index
        # self.unrouted = set(r for r in instance.requests if instance.carrier_assignment(r) == carrier_index)
        self.unrouted_requests = set()
        self.tours: List[tr.Tour] = []

        self.cost = 0
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
            'sum_travel_distance': self.sum_travel_distance(),
            'sum_travel_duration': self.sum_travel_duration(),
            'sum_load': self.sum_load(),
            'sum_revenue': self.sum_revenue(),
            'tour_summaries': {t.id_: t.summary() for t in self.tours}
        }


def read_solution_and_summary_from_json(path: Path):
    with open(path, mode='r') as f:
        solution, summary = json.load(f).values()
    return solution, summary
