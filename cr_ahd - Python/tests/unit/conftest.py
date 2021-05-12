import datetime as dt
from collections import Callable

import pytest
from pathlib import Path

import src.cr_ahd.utility_module.utils
import src.cr_ahd.utility_module.utils as ut
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module.utils import n_points_on_a_circle


@pytest.fixture()
def inst_gh_0():
    """
    a small custom instance based on the format of Gansterer & Hartl designed specifically for testing
    """
    return it.read_gansterer_hartl_mv(Path('./fixtures/test=1+dist=200+rad=200+n=5.dat'))


@pytest.fixture()
def sol_gh_0(inst_gh_0):
    return slt.GlobalSolution(inst_gh_0)


@pytest.fixture(scope='function', name='inst_and_sol_gh_0_ass9')
def inst_and_sol_gh_0_ass9(inst_gh_0: it.PDPInstance, sol_gh_0: slt.GlobalSolution):
    instance = inst_gh_0
    solution = sol_gh_0
    solution.assign_requests_to_carriers([0, 1, 2, 5, 6, 7, 10, 11, 12], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    return instance, solution


@pytest.fixture(scope='function')
def inst_and_sol_gh_0_ass9_routed6(inst_and_sol_gh_0_ass9):
    instance, solution = inst_and_sol_gh_0_ass9
    # carrier 0
    carrier = 0
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carriers[carrier].tours.append(tour)
    for request in solution.carriers[carrier].unrouted_requests[:2]:
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        solution.carriers[carrier].unrouted_requests.remove(request)

    # carrier 1
    carrier = 1
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carriers[carrier].tours.append(tour)
    tour.insert_and_update(instance, solution, [1, 2, 3, 4], [9, 8, 24, 23])
    solution.carriers[carrier].unrouted_requests.remove(5)
    solution.carriers[carrier].unrouted_requests.remove(6)

    # carrier 2
    carrier = 2
    tour = tr.Tour(0, instance, solution, carrier)
    solution.carriers[carrier].tours.append(tour)
    for request in solution.carriers[carrier].unrouted_requests[:2]:
        tour.insert_and_update(instance, solution, [1, 2], instance.pickup_delivery_pair(request))
        solution.carriers[carrier].unrouted_requests.remove(request)
    return instance, solution


@pytest.fixture
def inst_sol_pool_gh_0_ass9_routed6(inst_and_sol_gh_0_ass9_routed6):
    instance, solution = inst_and_sol_gh_0_ass9_routed6
    # put all unrouted requests into the auction pool
    pool = []
    for carrier_ in solution.carriers:
        pool.extend(carrier_.unrouted_requests)
    return instance, solution, pool


@pytest.fixture
def inst_sol_bundles_gh_0_ass9_routed6(inst_sol_pool_gh_0_ass9_routed6):
    instance, solution, pool = inst_and_sol_gh_0_ass9_routed6
    # create bundles from the pool
    raise NotImplementedError


@pytest.fixture
def instance_generator_circular():
    def _instance_generator(num_carriers: int = 3,
                            num_requests_per_carrier: int = 4,
                            max_num_tours_per_carrier: int = 3,
                            max_tour_length: float = 1200,
                            max_vehicle_load: float = 100,
                            depot_radius=100,
                            requests_radius=50):
        carrier_depots_x, carrier_depots_y = zip(*n_points_on_a_circle(num_carriers, depot_radius))
        requests_pickup_x = []
        requests_pickup_y = []
        requests_delivery_x = []
        requests_delivery_y = []
        for i in range(num_carriers):
            xs, ys = zip(*n_points_on_a_circle(num_requests_per_carrier * 2, requests_radius,
                                               carrier_depots_x[i], carrier_depots_y[i]))
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


@pytest.fixture
def inst_circular_c3_n4_k3_L1200_Q100(instance_generator_circular):
    return instance_generator_circular()


@pytest.fixture(name='sol_circular_c3_n4_k3_L1200_Q100')
def sol_circular_c3_n4_k3_L1200_Q100(inst_circular_c3_n4_k3_L1200_Q100):
    return slt.GlobalSolution(inst_circular_c3_n4_k3_L1200_Q100)


@pytest.fixture
def inst_sol_circular_c3_n4_k3_L1200_Q100_ass6(inst_circular_c3_n4_k3_L1200_Q100: it.PDPInstance,
                                               sol_circular_c3_n4_k3_L1200_Q100: slt.GlobalSolution):
    inst = inst_circular_c3_n4_k3_L1200_Q100
    sol = sol_circular_c3_n4_k3_L1200_Q100
    sol.assign_requests_to_carriers([0, 1, 4, 5, 8, 9], [0, 0, 1, 1, 2, 2])
    return inst, sol


@pytest.fixture
def inst_sol_circular_ass_routed_generator(instance_generator_circular: Callable):
    def _inst_sol_circular_ass_routed_generator(num_carriers: int = 3, num_requests_per_carrier: int = 4,
                                                max_num_tours_per_carrier: int = 3, max_tour_length: float = 1200,
                                                max_vehicle_load: float = 100, depot_radius: int = 100,
                                                requests_radius: int = 50, num_ass_per_carrier: int = 2,
                                                num_routed_per_carrier: int = 1):
        inst: it.PDPInstance = instance_generator_circular(num_carriers=num_carriers,
                                                           num_requests_per_carrier=num_requests_per_carrier,
                                                           max_num_tours_per_carrier=max_num_tours_per_carrier,
                                                           max_tour_length=max_tour_length,
                                                           max_vehicle_load=max_vehicle_load,
                                                           depot_radius=depot_radius,
                                                           requests_radius=requests_radius
                                                           )
        sol = slt.GlobalSolution(inst)
        for carrier in range(num_carriers):
            # assign requests
            ass = range(carrier * num_requests_per_carrier, carrier * num_requests_per_carrier + num_ass_per_carrier)
            sol.assign_requests_to_carriers(ass, [carrier] * num_ass_per_carrier)

            # route the assigned requests' vertices
            tour_ = tr.Tour(0, inst, sol, carrier)
            sol.carriers[carrier].tours.append(tour_)
            routed_vertices = []
            for r in ass[:num_routed_per_carrier]:
                routed_vertices.extend(inst.pickup_delivery_pair(r))
                sol.carriers[carrier].unrouted_requests.remove(r)

            ins_indices = range(1, num_routed_per_carrier * 2 + 1)

            # carrier 1: all pickup first, then all delivery
            if carrier == 1:
                routed_vertices = sorted(routed_vertices)

            tour_.insert_and_update(inst, sol, ins_indices, routed_vertices)

        return inst, sol

    return _inst_sol_circular_ass_routed_generator


@pytest.fixture
def inst_sol_pool_circular_c3_n6_k3_L1200_Q100_ass4_routed_2(inst_sol_circular_ass_routed_generator: Callable):
    instance, solution = inst_sol_circular_ass_routed_generator(num_carriers=3,
                                                                num_requests_per_carrier=6,
                                                                max_num_tours_per_carrier=3,
                                                                max_tour_length=1200,
                                                                max_vehicle_load=100,
                                                                num_ass_per_carrier=4,
                                                                num_routed_per_carrier=2,
                                                                depot_radius=100,
                                                                requests_radius=120,
                                                                )
    # put all unrouted requests into the auction pool
    pool = []
    for carrier_ in solution.carriers:
        pool.extend(carrier_.unrouted_requests)
    return instance, solution, pool


@pytest.fixture
def inst_sol_bundles_circular_c3_n6_k3_L1200_Q100_ass4_routed_2(
        inst_sol_pool_circular_c3_n6_k3_L1200_Q100_ass4_routed_2):
    instance, solution, pool = inst_sol_pool_circular_c3_n6_k3_L1200_Q100_ass4_routed_2
    raise NotImplementedError


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
                    Vertex(f'r{i - 1}', x, y, 0, ut.START_TIME + i * src.cr_ahd.utility_module.utils.TW_LENGTH,
                           ut.START_TIME + (i + 1) * src.cr_ahd.utility_module.utils.TW_LENGTH, dt.timedelta(minutes=10)))
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        return requests[1:]

    return _request_vertices_spiral
