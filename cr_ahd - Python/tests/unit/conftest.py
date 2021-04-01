import datetime as dt

import numpy as np
import pytest
import random

import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.instance import Instance
from src.cr_ahd.core_module.tour import Tour
from src.cr_ahd.core_module.vehicle import Vehicle
from src.cr_ahd.core_module.vertex import DepotVertex, Vertex


# ==========
# TOURS
# ==========
@pytest.fixture
def small_criss_cross_tour(depot_vertex, request_vertices_a):
    distance_matrix = ut.make_travel_dist_matrix([depot_vertex, *request_vertices_a])
    tour = Tour('small_criss_cross_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_a):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tour(depot_vertex, request_vertices_b):
    distance_matrix = ut.make_travel_dist_matrix([depot_vertex, *request_vertices_b])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_b):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def small_tw_tour(depot_vertex, request_vertices_c):
    distance_matrix = ut.make_travel_dist_matrix([depot_vertex, *request_vertices_c])
    tour = Tour('small_tour', depot_vertex, distance_matrix)
    for index, request in enumerate(request_vertices_c):
        tour.insert_and_update(index + 1, request)
    return tour


@pytest.fixture
def spiral_tour(depot_vertex, request_vertices_spiral):
    def _spiral_tour(num_requests):
        spiral_size = np.ceil(np.sqrt(num_requests + 1)).astype(int)
        requests = request_vertices_spiral(spiral_size, spiral_size)
        distance_matrix = ut.make_travel_dist_matrix([*requests, depot_vertex])
        tour = Tour('spiral_tour', depot_vertex, distance_matrix)
        for index, request in enumerate(requests):
            tour.insert_and_update(index + 1, request)
        return tour

    return _spiral_tour


# ==========
# VERTICES
# ==========

@pytest.fixture
def depot_vertex():
    return DepotVertex('d0', 0, 0)


@pytest.fixture
def request_vertices_a():
    """criss-cross sequence. use to test 2opt"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_b():
    """used to test reversing part of a tour"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_c():
    """used to test whether tour.reverse_section() handles InsertionError correctly by going back to the original"""
    return [
        Vertex('r0', 0, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 20, 0, dt.datetime.min, dt.datetime(year=1, month=1, day=1, hour=0, minute=46)),
        # infeasible if positioned at routing_sequence[5]
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_d():
    """use to test HighestInsertionCostDistance request selection"""
    return [
        Vertex('r0', 10, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 20, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_e():
    """2 clusters of 4"""
    return [
        Vertex('r0', 5, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 20, 5, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 10, 25, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r6', 25, 5, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),

        Vertex('r3', 15, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', 10, 15, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r5', 20, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r7', 25, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_f():
    """testing travel times and distances"""
    return [
        DepotVertex('d0', 0, 0),
        Vertex('r0', -10, 0, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r1', -10, -10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r2', 10, -10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r3', 10, 10, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
        Vertex('r4', 0, 20, 0, dt.datetime.min, dt.datetime.max - dt.timedelta(microseconds=1)),
    ]


@pytest.fixture
def request_vertices_g():
    """all different time windows"""
    return [
        Vertex('r0', -10, 0, 0, dt.datetime(1, 1, 1, hour=8), dt.datetime(1, 1, 1, hour=10)),
        Vertex('r1', -10, -10, 0, dt.datetime(1, 1, 1, hour=10), dt.datetime(1, 1, 1, hour=12)),
        Vertex('r2', 10, -10, 0, dt.datetime(1, 1, 1, hour=12), dt.datetime(1, 1, 1, hour=14)),
        Vertex('r3', 10, 10, 0, dt.datetime(1, 1, 1, hour=6), dt.datetime(1, 1, 1, hour=8)),
        Vertex('r4', 0, 20, 0, dt.datetime(1, 1, 1, hour=16), dt.datetime(1, 1, 1, hour=18)),
    ]


@pytest.fixture
def request_vertices_h():
    return [
        Vertex('r0', 0, 120, 0, dt.datetime(1, 1, 1, hour=8), dt.datetime(1, 1, 1, hour=10)),
        Vertex('r1', 120, 120, 0, dt.datetime(1, 1, 1, hour=10), dt.datetime(1, 1, 1, hour=12)),
        Vertex('r2', 120, 0, 0, dt.datetime(1, 1, 1, hour=12), dt.datetime(1, 1, 1, hour=14)),
        Vertex('r3', 120, -120, 0, dt.datetime(1, 1, 1, hour=14), dt.datetime(1, 1, 1, hour=16)),
        Vertex('r4', 0, -120, 0, dt.datetime(1, 1, 1, hour=16), dt.datetime(1, 1, 1, hour=18)),
    ]


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


@pytest.fixture
def request_vertices_random_6():
    """collection of 6 randomly placed vertices [0, 100] with equal tw (0, max) and no service time"""
    return [Vertex(f'r{i}', np.random.randint(0, 100), np.random.randint(0, 100), 0, dt.datetime.min,
                   dt.datetime.max - dt.timedelta(microseconds=1))
            for i in range(6)]


@pytest.fixture
def request_vertices_random_15():
    """collection of 6 randomly placed vertices [0, 100] with equal tw (0, max) and no service time"""
    return [Vertex(f'r{i}', np.random.randint(0, 100), np.random.randint(0, 100), 0, dt.datetime.min,
                   dt.datetime.max - dt.timedelta(microseconds=1))
            for i in range(15)]


# ==========
# VEHICLES
# ==========
@pytest.fixture
def vehicles_capacity_0():
    def _vehicles_capacity_0(num_vehicles):
        return [Vehicle(f'v{i}', 0) for i in range(num_vehicles)]

    return _vehicles_capacity_0


# ==========
# CARRIERS
# ==========

@pytest.fixture
def carrier_a(depot_vertex, vehicles_capacity_0, request_vertices_d):
    dist_matrix = ut.make_travel_dist_matrix([*request_vertices_d, depot_vertex])
    return Carrier('c0', depot_vertex, vehicles_capacity_0(5), request_vertices_d, dist_matrix)


@pytest.fixture
def carrier_a_partially_routed(carrier_a):
    for request in carrier_a.requests[:3]:
        carrier_a.vehicles[0].tour.insert_and_update(1, request)
    return carrier_a


@pytest.fixture
def carrier_b(depot_vertex, vehicles_capacity_0, request_vertices_d):
    dist_matrix = ut.make_travel_dist_matrix([*request_vertices_d, depot_vertex])
    return Carrier('c1', depot_vertex, vehicles_capacity_0(5), request_vertices_d, dist_matrix)


@pytest.fixture
def carriers_and_unassigned_requests_3_6(depot_vertex, vehicles_capacity_0, request_vertices_random_6):
    dist_matrix = ut.make_travel_dist_matrix([*request_vertices_random_6, depot_vertex])
    vehicles = vehicles_capacity_0(15)
    carriers = []
    for i in range(3):
        carriers.append(
            Carrier(f'c{i}', depot_vertex, vehicles[i * 5:(i + 1) * 5], dist_matrix=dist_matrix))
    return carriers, request_vertices_random_6


@pytest.fixture
def carrier_c(depot_vertex, vehicles_capacity_0, request_vertices_g):
    dist_matrix = ut.make_travel_dist_matrix([*request_vertices_g, depot_vertex])
    return Carrier('c1', depot_vertex, vehicles_capacity_0(5), request_vertices_g, dist_matrix)


@pytest.fixture()
def carrier_spiral_partially_routed(depot_vertex, request_vertices_spiral, vehicles_capacity_0):
    def _carrier_spiral_partially_routed(request_spiral_size: int, num_vehicles: int, num_routed_requests: int):
        requests = request_vertices_spiral(request_spiral_size, request_spiral_size)
        dist_matrix = ut.make_travel_dist_matrix([*requests, depot_vertex])
        carrier = Carrier('c0', depot_vertex, vehicles_capacity_0(num_vehicles), requests, dist_matrix)
        tour = carrier.vehicles[0].tour
        for i in range(num_routed_requests):
            tour.insert_and_update(i + 1, carrier.unrouted_requests[0])
        return carrier

    return _carrier_spiral_partially_routed


# ==========
# SUBMITTED REQUESTS
# ==========

@pytest.fixture
def submitted_requests_random_6(request_vertices_random_6):
    length = round(len(request_vertices_random_6) / 2)
    return dict(c1=request_vertices_random_6[:length], c2=request_vertices_random_6[length:])


@pytest.fixture
def submitted_requests_random_15(request_vertices_random_15):
    split = list(ut.split_iterable(request_vertices_random_15, 3))
    return dict(c0=split[0], c1=split[1], c2=split[2])


@pytest.fixture
def submitted_requests_a(request_vertices_e):
    split = list(ut.split_iterable(request_vertices_e, 2))
    return dict(c0=split[0], c1=split[1])


# ==========
# BUNDLE SETS
# ==========

@pytest.fixture
def bundle_set_a(request_vertices_a, depot_vertex, vehicles_capacity_0):
    distance_matrix = ut.make_travel_dist_matrix([*request_vertices_a, depot_vertex])
    carrier = Carrier('test', depot_vertex, vehicles_capacity_0(5), dist_matrix=distance_matrix)
    bundle_set = {0: request_vertices_a[:2],
                  1: request_vertices_a[2:3],
                  2: request_vertices_a[3:]}
    return carrier, bundle_set


# ==========
# BIDS
# ==========

@pytest.fixture
def bids_a(carriers_and_unassigned_requests_3_6):
    """random bids on some random bundles. does certainly not represent distance insertion cost!

    :return: dict of bids per bundle per carrier: \n {bundleA: {carrier1: bid, carrier2: bid}, bundleB: {carrier1: bid,
        carrier2: bid} \n Dict[Tuple[Vertex], Dict[Carrier, float]]"""
    carriers, requests = carriers_and_unassigned_requests_3_6
    bids = {
        tuple(requests[:2]): {
            carriers[0]: 9,
            carriers[1]: 7,
            carriers[2]: 5,  # lowest
        },
        tuple(requests[2:4]): {
            carriers[0]: 1,  # lowest
            carriers[1]: 2,
            carriers[2]: 4,
        },
        tuple(requests[4:6]): {
            carriers[0]: 10,
            carriers[1]: 7,  # lowest
            carriers[2]: 11,
        },
    }
    return bids


# ==========
# INSTANCES
# ==========
@pytest.fixture
def instance_a(depot_vertex, carriers_and_unassigned_requests_3_6):
    carriers, requests = carriers_and_unassigned_requests_3_6
    for i, r in enumerate(requests):
        r.carrier_assignment = carriers[int(i / 2)].id_
    return Instance('instance_a', requests, carriers, carriers[0].distance_matrix)


# ==========
# TIME WINDOWS
# ==========

@pytest.fixture
def tw_hour():
    def _tw_hour(from_hour, to_hour):
        return ut.TimeWindow(dt.datetime.min + dt.timedelta(hours=from_hour),
                             dt.datetime.min + dt.timedelta(hours=to_hour))

    return _tw_hour


@pytest.fixture
def tw_offer_set():
    def _tw_offer_set(num_tw: int, tw_length: dt.timedelta, every_other: bool, shuffle: bool, start_time=ut.START_TIME,
                      end_time=ut.END_TIME):
        """

        :param num_tw: How many time windows should be generated?
        :param tw_length: duration of each of the time windows
        :param every_other: sample only the 'even' time windows from the list. will half the num_tw
        :param shuffle: shall the order of the tw be shuffled?
        :param start_time: the beginning of the first tw
        :param end_time: the end of the latest tw
        :return:
        """
        offer_set = list(ut.datetime_range(start_time, end_time, freq=tw_length))[:num_tw]
        offer_set = [ut.TimeWindow(e, e + tw_length) for e in offer_set]
        if every_other:
            offer_set = [x for i, x in enumerate(offer_set) if i % 2 == 0]
        if shuffle:
            random.shuffle(offer_set)
        return offer_set

    return _tw_offer_set
