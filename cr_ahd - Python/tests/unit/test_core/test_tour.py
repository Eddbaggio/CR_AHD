from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr
from src.cr_ahd.utility_module import utils as ut
import pytest
import math
import datetime as dt


# def test_removal_distance_delta(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.GlobalSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carrier_solutions[c].tours[t]
#     delta = tour.removal_distance_delta(instance, [1])
#     assert delta == 50

# def test__insert_no_update(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.CAHDSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carriers[c].tours[t]
#     tour._multi_insert_no_update(1, 5)
#     assert tour.routing_sequence == (0, 5, 4, 19, 3, 18, 0)
    # assert tour.revenue_sequence == ()
    # assert tour.load_sequence == ()
    # assert tour.travel_distance_sequence == ()
    # assert tour.travel_duration_sequence == ()


# def test_insertion_distance_delta(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.GlobalSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carrier_solutions[c].tours[t]
#     delta = tour.single_insertion_distance_delta(instance, [1, 6], [5, 20])
#     assert delta == pytest.approx(200 - math.sqrt(25 ** 2 + 50 ** 2) - math.sqrt(100 ** 2 + 25 ** 2))


# def test_insert_and_update(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.CAHDSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carriers[c].tours[t]
#     distance_before = tour.sum_travel_distance
#     tour.multi_insert_and_update([1, 6],, [5, 20],
#     assert tour.routing_sequence == (0, 5, 4, 19, 3, 18, 20, 0)
#     assert tour.sum_travel_distance == pytest.approx(
#         distance_before + 200 - math.sqrt(25 ** 2 + 50 ** 2) - math.sqrt(100 ** 2 + 25 ** 2))


# def test__pop_no_update(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.CAHDSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carriers[c].tours[t]
#     tour._pop_no_update(3)
#     assert tour.routing_sequence == (0, 4, 19, 18, 0)
    # assert tour.revenue_sequence == ()
    # assert tour.load_sequence == ()
    # assert tour.travel_distance_sequence == ()
    # assert tour.travel_duration_sequence == ()


# def test__pop_and_update(inst_and_sol_gh_0_ass9_routed6):
#     instance: it.PDPInstance
#     solution: slt.CAHDSolution
#     instance, solution = inst_and_sol_gh_0_ass9_routed6
#     c = 0
#     t = 0
#     tour = solution.carriers[c].tours[t]
#     travel_dist_before = tour.sum_travel_distance
#     tour.pop_and_update(instance, solution, [1, 2],,
#     assert tour.routing_sequence == (0, 3, 18, 0)
#     assert tour.sum_travel_distance == pytest.approx(
#         travel_dist_before - math.sqrt(25 ** 2 + 50 ** 2) - 50 - math.sqrt(50 ** 2 + 50 ** 2) + math.sqrt(
#             25 ** 2 + 100 ** 2))


@pytest.fixture
def vansteenwegen_example_insert():
    h = dict(tw_open=dt.datetime(1, 1, 1, 0, 0),
             tw_close=dt.datetime(1, 1, 1, 0, 5),
             service_duration=dt.timedelta(minutes=4),
             arrival=dt.datetime(1, 1, 1, 0, 0),
             service=dt.datetime(1, 1, 1, 0, 0),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=5)
             )
    i = dict(tw_open=dt.datetime(1, 1, 1, 0, 1),
             tw_close=dt.datetime(1, 1, 1, 0, 33),
             service_duration=dt.timedelta(minutes=5),
             arrival=dt.datetime(1, 1, 1, 0, 5),
             service=dt.datetime(1, 1, 1, 0, 5),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=17)
             )
    j = dict(tw_open=dt.datetime(1, 1, 1, 0, 13),
             tw_close=dt.datetime(1, 1, 1, 0, 31),
             service_duration=dt.timedelta(minutes=4),
             arrival=None,
             service=None,
             wait=None,
             max_shift=None
             )
    k = dict(tw_open=dt.datetime(1, 1, 1, 0, 17),
             tw_close=dt.datetime(1, 1, 1, 0, 36),
             service_duration=dt.timedelta(minutes=8),
             arrival=dt.datetime(1, 1, 1, 0, 13),
             service=dt.datetime(1, 1, 1, 0, 17),
             wait=dt.timedelta(minutes=4),
             max_shift=dt.timedelta(minutes=13)
             )
    l = dict(tw_open=dt.datetime(1, 1, 1, 0, 30),
             tw_close=dt.datetime(1, 1, 1, 0, 40),
             service_duration=dt.timedelta(minutes=5),
             arrival=dt.datetime(1, 1, 1, 0, 27),
             service=dt.datetime(1, 1, 1, 0, 30),
             wait=dt.timedelta(minutes=3),
             max_shift=dt.timedelta(minutes=10)
             )
    nodes = [h, i, j, k, l]
    routed = [h, i, k, l]

    data = dict(
        routing_sequence=[0, 1, 3, 4],
        sum_travel_distance=6,
        sum_travel_duration=ut.travel_time(6),
        sum_revenue=0,
        sum_profit=-6,
        arrival_schedule=[x['arrival'] for x in routed],
        service_schedule=[x['service'] for x in routed],
        sum_load=0,
        wait_sequence=[x['wait'] for x in routed],
        max_shift_sequence=[x['max_shift'] for x in routed],
        num_depots=1,
        num_requests=20,  # some high number to avoid precedence feasibility checks
        distance_matrix=[
           # h, i, j, k, l
            [0, 1,         ],  # h
            [1, 0, 1, 3,   ],  # i
            [0, 1, 0, 3,   ],  # j
            [0, 0, 3, 0, 2 ],  # k
            [0, 0, 0, 2, 0 ],  # l
                         ],
        vertex_load=[0, 0, 0, 0, 0],
        revenue=[0, 0, 0, 0, 0],
        service_duration=[x['service_duration'] for x in nodes],
        vehicles_max_travel_distance=100,
        vehicles_max_load=100,
        tw_open=[x['tw_open'] for x in nodes],
        tw_close=[x['tw_close'] for x in nodes],
        insertion_index=2,
        insertion_vertex=2,
    )
    return data


def test_single_insertion_feasibility_check(vansteenwegen_example_insert):
    """
    Example taken from Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local
    search for the team orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
    https://doi.org/10.1016/j.cor.2009.03.008

    """

    assert tr.single_insertion_feasibility_check(**vansteenwegen_example_insert)


def test_single_insert_and_update(vansteenwegen_example_insert):

    updated_input = tr.single_insert_and_update(**vansteenwegen_example_insert)
    assert vansteenwegen_example_insert['wait_sequence'] == [dt.timedelta(minutes=x) for x in [0, 0, 2, 0, 0]]
    assert vansteenwegen_example_insert['max_shift_sequence'] == [dt.timedelta(minutes=x) for x in [5, 12, 10, 10, 10]]
    assert vansteenwegen_example_insert['arrival_schedule'] == [dt.datetime(1, 1, 1, 0, x) for x in [0, 5, 11, 20, 30]]
    assert vansteenwegen_example_insert['service_schedule'] == [dt.datetime(1, 1, 1, 0, x) for x in [0, 5, 13, 20, 30]]

@pytest.fixture
def vansteenwegen_example_pop():
    h = dict(tw_open=dt.datetime(1, 1, 1, 0, 0),
             tw_close=dt.datetime(1, 1, 1, 0, 5),
             service_duration=dt.timedelta(minutes=4),
             arrival=dt.datetime(1, 1, 1, 0, 0),
             service=dt.datetime(1, 1, 1, 0, 0),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=5)
             )
    i = dict(tw_open=dt.datetime(1, 1, 1, 0, 1),
             tw_close=dt.datetime(1, 1, 1, 0, 33),
             service_duration=dt.timedelta(minutes=5),
             arrival=dt.datetime(1, 1, 1, 0, 5),
             service=dt.datetime(1, 1, 1, 0, 5),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=12)
             )
    j = dict(tw_open=dt.datetime(1, 1, 1, 0, 13),
             tw_close=dt.datetime(1, 1, 1, 0, 31),
             service_duration=dt.timedelta(minutes=4),
             arrival=dt.datetime(1, 1, 1, 0, 13),
             service=dt.datetime(1, 1, 1, 0, 13),
             wait=dt.timedelta(minutes=2),
             max_shift=dt.timedelta(minutes=10)
             )
    k = dict(tw_open=dt.datetime(1, 1, 1, 0, 17),
             tw_close=dt.datetime(1, 1, 1, 0, 36),
             service_duration=dt.timedelta(minutes=8),
             arrival=dt.datetime(1, 1, 1, 0, 20),
             service=dt.datetime(1, 1, 1, 0, 20),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=10)
             )
    l = dict(tw_open=dt.datetime(1, 1, 1, 0, 30),
             tw_close=dt.datetime(1, 1, 1, 0, 40),
             service_duration=dt.timedelta(minutes=5),
             arrival=dt.datetime(1, 1, 1, 0, 30),
             service=dt.datetime(1, 1, 1, 0, 30),
             wait=dt.timedelta(minutes=0),
             max_shift=dt.timedelta(minutes=10)
             )
    nodes = [h, i, j, k, l]
    routed = [h, i, j, k, l]

    data = dict(
        routing_sequence=[0, 1, 2, 3, 4],
        sum_travel_distance=7,
        sum_travel_duration=ut.travel_time(6),
        sum_revenue=0,
        sum_profit=-7,
        arrival_schedule=[x['arrival'] for x in routed],
        service_schedule=[x['service'] for x in routed],
        sum_load=0,
        wait_sequence=[x['wait'] for x in routed],
        max_shift_sequence=[x['max_shift'] for x in routed],
        distance_matrix=[
           # h, i, j, k, l
            [0, 1,         ],  # h
            [1, 0, 1, 3,   ],  # i
            [0, 1, 0, 3,   ],  # j
            [0, 0, 3, 0, 2 ],  # k
            [0, 0, 0, 2, 0 ],  # l
                         ],
        vertex_load=[0, 0, 0, 0, 0],
        revenue=[0, 0, 0, 0, 0],
        service_duration=[x['service_duration'] for x in nodes],
        tw_open=[x['tw_open'] for x in nodes],
        tw_close=[x['tw_close'] for x in nodes],
        pop_pos=2,
    )
    return data


def test_single_pop_and_update(vansteenwegen_example_pop):
    popped, updated_sums = tr.single_pop_and_update(**vansteenwegen_example_pop)
    assert popped == 2
    assert vansteenwegen_example_pop['wait_sequence'] == [dt.timedelta(minutes=x) for x in [0, 0, 4, 3]]
    assert vansteenwegen_example_pop['max_shift_sequence'] == [dt.timedelta(minutes=x) for x in [5, 17, 13, 10]]
    assert vansteenwegen_example_pop['arrival_schedule'] == [dt.datetime(1, 1, 1, 0, x) for x in [0, 5, 13, 27]]
    assert vansteenwegen_example_pop['service_schedule'] == [dt.datetime(1, 1, 1, 0, x) for x in [0, 5, 17, 30]]
