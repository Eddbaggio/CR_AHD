import datetime as dt

import numpy as np
import math
import pytest
import random
from pathlib import Path
import pandas as pd

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr


@pytest.fixture(scope='class')
def inst_gh_0():
    """
    a small custom instance based on the format of Gansterer & Hartl designed specifically for testing
    """
    return it.read_gansterer_hartl_mv(Path('./fixtures/test=1+dist=200+rad=200+n=5.dat'))


@pytest.fixture(scope='class')
def sol_gh_0(inst_gh_0):
    return slt.GlobalSolution(inst_gh_0)


@pytest.fixture
def inst_and_sol_gh_0_9_assigned(inst_gh_0: it.PDPInstance, sol_gh_0: slt.GlobalSolution):
    instance = inst_gh_0
    solution = sol_gh_0
    solution.assign_requests_to_carriers([0, 1, 2, 5, 6, 7, 10, 11, 12], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    return instance, solution


@pytest.fixture
def inst_and_sol_gh_0_9_ass_6_routed(inst_and_sol_gh_0_9_assigned):
    instance, solution = inst_and_sol_gh_0_9_assigned
    # carrier 0
    carrier = 0
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carrier_solutions[carrier].tours.append(tour)
    for request in solution.carrier_solutions[carrier].unrouted_requests[:2]:
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        solution.carrier_solutions[carrier].unrouted_requests.remove(request)

    # carrier 1
    carrier = 1
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carrier_solutions[carrier].tours.append(tour)
    tour.insert_and_update(instance, solution, [1, 2, 3, 4], [9, 8, 24, 23])
    solution.carrier_solutions[carrier].unrouted_requests.remove(5)
    solution.carrier_solutions[carrier].unrouted_requests.remove(6)

    # carrier 2
    carrier = 2
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carrier_solutions[carrier].tours.append(tour)
    for request in solution.carrier_solutions[carrier].unrouted_requests[:2]:
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        solution.carrier_solutions[carrier].unrouted_requests.remove(request)
    return instance, solution


@pytest.fixture
def instance_generator_circular():
    def _instance_generator(num_carriers: int = 3,
                            num_requests_per_carrier: int = 4,
                            max_num_tours_per_carrier: int = 3,
                            max_tour_length: float = 1200,
                            max_vehicle_load: float = 100):
        carrier_depots_x, carrier_depots_y = zip(*n_points_on_a_circle(num_carriers, 100))
        requests_pickup_x = []
        requests_pickup_y = []
        requests_delivery_x = []
        requests_delivery_y = []
        for i in range(num_carriers):
            xs, ys = zip(
                *n_points_on_a_circle(num_requests_per_carrier * 2, 50, carrier_depots_x[i], carrier_depots_y[i]))
            for j, (x, y) in enumerate(zip(xs, ys)):
                if j % 2 == 0:
                    requests_pickup_x.append(x)
                    requests_pickup_y.append(y)
                else:
                    requests_delivery_x.append(x)
                    requests_delivery_y.append(y)

        return it.PDPInstance(
            f'fixture=0+rad=100+n={num_requests_per_carrier}',
            max_num_tours_per_carrier=max_num_tours_per_carrier,
            max_vehicle_load=max_vehicle_load,
            max_tour_length=max_tour_length,
            requests=range(num_requests_per_carrier * num_carriers),
            requests_initial_carrier_assignment=ut.flatten(
                [[i] * num_requests_per_carrier for i in range(num_carriers)]),
            requests_pickup_x=requests_pickup_x,
            requests_pickup_y=requests_pickup_y,
            requests_delivery_x=requests_delivery_x,
            requests_delivery_y=requests_delivery_y,
            requests_revenue=[10] * num_carriers * num_requests_per_carrier,
            requests_pickup_service_time=[0] * num_carriers * num_requests_per_carrier,
            requests_delivery_service_time=[0] * num_carriers * num_requests_per_carrier,
            requests_pickup_load=[0] * num_carriers * num_requests_per_carrier,
            requests_delivery_load=[10] * num_carriers * num_requests_per_carrier,
            carrier_depots_x=carrier_depots_x,
            carrier_depots_y=carrier_depots_y,
        )

    return _instance_generator


def n_points_on_a_circle(n: int, radius, origin_x=0, origin_y=0):
    """create coordinates for n points that are evenly spaced on the circumference of  a circle of the given radius"""
    points = []
    for i in range(1, n + 1):
        x = radius * math.cos(2 * math.pi * i / n - math.pi / 2)
        y = radius * math.sin(2 * math.pi * i / n - math.pi / 2)
        points.append((origin_x + x, origin_y + y))
    return points


@pytest.fixture
def request_vertices_spiral():
    """https://stackoverflow.com/questions/398299/looping-in-a-spiral"""

    def _request_vertices_spiral(X, Y):
        requests = []
        x = y = 0
        dx = 0
        dy = -1
        for i in range(max(X, Y) ** 2):
            if (-X / 2 < x <= X / 2) and (-Y / 2 < y <= Y / 2):
                requests.append(
                    Vertex(f'r{i - 1}', x, y, 0, ut.START_TIME + i * ut.TW_LENGTH,
                           ut.START_TIME + (i + 1) * ut.TW_LENGTH, dt.timedelta(minutes=10)))
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        return requests[1:]

    return _request_vertices_spiral
