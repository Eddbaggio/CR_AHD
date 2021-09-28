import sys
import datetime as dt
import itertools
import math
import random
import re
from collections import namedtuple
from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import trange

from tw_management_module.tw import TimeWindow

Coordinates = namedtuple('Coords', ['x', 'y'])

# alpha 100%
univie_colors_100 = [
    '#0063A6',  # universitätsblau
    '#666666',  # universtitätsgrau
    '#A71C49',  # weinrot
    '#DD4814',  # orangerot
    '#F6A800',  # goldgelb
    '#94C154',  # hellgrün
    '#11897A',  # mintgrün
]
# lightness/alpha 60%
univie_colors_60 = [
    '#6899CA',  # universitätsblau
    '#B5B4B4',  # universtitätsgrau
    '#C26F76',  # weinrot
    '#F49C6A',  # orangerot
    '#FCCB78',  # goldgelb
    '#C3DC9F',  # hellgrün
    '#85B6AE',  # mintgrün
]

# paired
univie_colors_paired = list(itertools.chain(*zip(univie_colors_100, univie_colors_60)))

univie_cmap = LinearSegmentedColormap.from_list('univie', univie_colors_100, N=len(univie_colors_100))
univie_cmap_paired = LinearSegmentedColormap.from_list('univie_paired', univie_colors_paired,
                                                       N=len(univie_colors_100) + len(univie_colors_60))


def map_to_univie_colors(categories: Sequence):
    colormap = {}
    for cat, color in zip(categories, itertools.cycle(univie_colors_100 + univie_colors_60)):
        colormap[cat] = color
    return colormap


def split_iterable(iterable, num_chunks):
    """ splits an iterable, e.g. a list into num_chunks parts of roughly the same length. If no exact split is
    possible the first chunk(s) will be longer. """
    k, m = divmod(len(iterable), num_chunks)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks))


def _euclidean_distance(a: Coordinates, b: Coordinates):
    raise DeprecationWarning(f'Use new _euclidean_distance function!')
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def travel_time(dist):
    return dt.timedelta(hours=dist / SPEED_KMH)  # compute timedelta


def power_set(iterable, include_empty_set=True):
    """power_set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    if include_empty_set:
        rng = range(len(s) + 1)
    else:
        rng = range(1, len(s) + 1)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in rng)


def flatten(sequence: (List, Tuple)):
    if not sequence:
        return sequence
    if isinstance(sequence[0], (List, Tuple)):
        return flatten(sequence[0]) + flatten(sequence[1:])
    return sequence[:1] + flatten(sequence[1:])


def random_partition(li):
    min_chunk = 1
    max_chunk = len(li)
    it = iter(li)
    while True:
        randint = np.random.randint(min_chunk, max_chunk)
        nxt = list(itertools.islice(it, randint))
        if nxt:
            yield nxt
        else:
            break


def random_max_k_partition_idx(ls, max_k) -> List[int]:
    if max_k < 1:
        return []
    # randomly determine the actual k
    k = random.randint(1, min(max_k, len(ls)))
    # We require that this list contains k different values, so we start by adding each possible different value.
    indices = list(range(k))
    # now we add random values from range(k) to indices to fill it up to the length of ls
    indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
    # shuffle the indices into a random order
    random.shuffle(indices)
    return indices


def random_max_k_partition(ls, max_k) -> Sequence[Sequence[int]]:
    """partition ls in at most k randomly sized disjoint subsets

    """
    # https://stackoverflow.com/a/45880095
    # we need to know the length of ls, so convert it into a list
    ls = list(ls)
    # sanity check
    if max_k < 1:
        return []
    # randomly determine the actual k
    k = random.randint(1, min(max_k, len(ls)))
    # Create a list of length ls, where each element is the index of
    # the subset that the corresponding member of ls will be assigned
    # to.
    #
    # We require that this list contains k different values, so we
    # start by adding each possible different value.
    indices = list(range(k))
    # now we add random values from range(k) to indices to fill it up
    # to the length of ls
    indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
    # shuffle the indices into a random order
    random.shuffle(indices)
    return indices
    # construct and return the random subset: sort the elements by
    # which subset they will be assigned to, and group them into sets
    partitions = []
    sorted_ = sorted(zip(indices, ls), key=lambda x: x[0])
    for index, subset in itertools.groupby(sorted_, key=lambda x: x[0]):
        partitions.append([x[1] for x in subset])
    return partitions


def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])


def argsmin(a, k: int):
    """
    returns the indices of the k min arguments in a
    """
    assert k > 0
    a_sorted = sorted(range(len(a)), key=lambda x: a[x])  # indices in increasing order of values
    return a_sorted[:k]


def argsmax(a, k: int):
    """
    returns the indices of the k max arguments in a
    """
    assert k > 0
    a_sorted = sorted(range(len(a)), key=lambda x: a[x], reverse=True)  # indices in decreasing order of values
    return a_sorted[:k]


def midpoint(instance, pickup_vertex, delivery_vertex):
    pickup_x, pickup_y = instance.x_coords[pickup_vertex], instance.y_coords[delivery_vertex]
    delivery_x, delivery_y = instance.x_coords[pickup_vertex], instance.y_coords[delivery_vertex]
    return (pickup_x + delivery_x) / 2, (pickup_y + delivery_y) / 2


def midpoint_(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def linear_interpolation(iterable: Sequence, new_min: float, new_max: float, old_min=None, old_max=None):
    """
    return the iterable re-scaled to the range between new_min and new_max.
    https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445

    """
    if old_min is None and old_max is None:
        old_min = min(iterable)
        old_max = max(iterable)
    return [((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min for x in iterable]


def n_points_on_a_circle(n: int, radius, origin_x=0, origin_y=0):
    """create coordinates for n points that are evenly spaced on the circumference of  a circle of the given radius"""
    points = []
    for i in range(1, n + 1):
        x = radius * math.cos(2 * math.pi * i / n - math.pi / 2)
        y = radius * math.sin(2 * math.pi * i / n - math.pi / 2)
        points.append((origin_x + x, origin_y + y))
    return points


def datetime_range(start: dt.datetime, end: dt.datetime, freq: dt.timedelta, include_end=True):
    """
    returns a generator object that yields datetime objects in the range from start to end in steps of freq.
    :param include_end: determines whether the specified end is included in the range
    :return:
    """
    return (start + x * freq for x in range(((end - start) // freq) + include_end))


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(str(s))]


def indices_to_nested_lists(indices: Sequence[int], elements: Sequence):
    nested_list = [[] for _ in range(max(indices) + 1)]
    for x, y in zip(elements, indices):
        nested_list[y].append(x)
    return nested_list


def validate_solution(instance, solution):
    assert solution.num_carriers() > 0

    for carrier_id in trange(len(solution.carriers), desc=f'Solution validation', disable=True):
        carrier = solution.carriers[carrier_id]
        assert len(carrier.unrouted_requests) == 0
        for tour in carrier.tours:
            assert tour is solution.tours[tour.id_]
            assert tour.routing_sequence[0] == carrier.id_
            assert tour.routing_sequence[-1] == carrier.id_

            assert tour.sum_load == 0, instance.id_
            assert tour.sum_travel_distance <= instance.vehicles_max_travel_distance, instance.id_
            assert round(tour.sum_profit, 4) == round(tour.sum_revenue - tour.sum_travel_distance, 4), \
                f'{instance.id_}: {round(tour.sum_profit, 4)}!={round(tour.sum_revenue - tour.sum_travel_distance, 4)}'

            validate_tour(instance, tour)

            # request-to-tour assignment record
            # for vertex in tour.routing_sequence[1:-1]:
            #     request = instance.request_from_vertex(vertex)
            #     assert solution.request_to_tour_assignment[
            #                request] == tour.id_, f'{instance.id_}, tour {tour.id_}, vertex {vertex} at index {i}'


def validate_tour(instance, tour):
    # iterate over the tour
    for i in trange(1, len(tour.routing_sequence), desc=f'Tour {tour.id_}', disable=True):
        predecessor = tour.routing_sequence[i - 1]
        vertex = tour.routing_sequence[i]
        msg = f'{instance.id_}, tour {tour.id_}, vertex {vertex} at index {i}'

        # routing and service time constraint
        assert tour.arrival_time_sequence[i] == tour.service_time_sequence[i - 1] + \
               instance.vertex_service_duration[predecessor] + \
               instance.travel_duration([predecessor], [vertex]), msg

        # waiting times
        assert tour.service_time_sequence[i] == tour.arrival_time_sequence[i] + \
               tour.wait_duration_sequence[i], msg
        assert tour.wait_duration_sequence[i] == max(
            dt.timedelta(0), instance.tw_open[vertex] - tour.arrival_time_sequence[i]), msg

        # max_shift times
        if instance.vertex_type(vertex) != 'depot':
            assert tour.max_shift_sequence[i] == min(
                instance.tw_close[vertex] - tour.service_time_sequence[i],
                tour.wait_duration_sequence[i + 1] + tour.max_shift_sequence[i + 1]
            ), msg
        else:
            assert tour.max_shift_sequence[i] == instance.tw_close[vertex] - tour.service_time_sequence[i]

        # tw constraint
        assert instance.tw_open[vertex] <= tour.service_time_sequence[i] <= instance.tw_close[vertex], \
            msg

        # precedence constraint
        if instance.vertex_type(vertex) == 'pickup':
            assert vertex + instance.num_requests in tour.routing_sequence[i:], msg
        elif instance.vertex_type(vertex) == 'delivery':
            assert vertex - instance.num_requests in tour.routing_sequence[:i], msg
        else:
            assert vertex in range(instance.num_carriers), msg

        # meta data
        if instance.vertex_type(vertex) != 'depot':
            assert tour.vertex_pos[vertex] == i, msg


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


random.seed(0)
DISTANCE_SCALING = 1
REVENUE_SCALING = DISTANCE_SCALING
LOAD_CAPACITY_SCALING = 10
START_TIME: dt.datetime = dt.datetime.min
END_TIME: dt.datetime = dt.datetime.min + dt.timedelta(minutes=3360)
# END_TIME = dt.datetime.min + dt.timedelta(days=1)
TW_LENGTH: dt.timedelta = dt.timedelta(hours=2)
ALL_TW = [TimeWindow(e, min(e + TW_LENGTH, END_TIME)) for e in
          datetime_range(START_TIME, END_TIME, freq=TW_LENGTH, include_end=False)]
TIME_HORIZON = TimeWindow(START_TIME, END_TIME)
SPEED_KMH = 60  # vehicle speed (set to 60 to treat distance = time)

solver_config = [
    'solution_algorithm',
    'tour_improvement',
    'neighborhoods',
    'tour_construction',
    'tour_improvement_time_limit_per_carrier',
    'time_window_offering',
    'time_window_selection',

    'num_int_auctions',
    'int_auction_tour_construction',
    'int_auction_tour_improvement',
    'int_auction_neighborhoods',
    'int_auction_num_submitted_requests',
    'int_auction_request_selection',
    'int_auction_bundle_generation',
    'int_auction_bundling_valuation',
    'int_auction_num_auction_bundles',
    'int_auction_bidding',
    'int_auction_winner_determination',
    'int_auction_num_auction_rounds',

    'fin_auction_tour_construction',
    'fin_auction_tour_improvement',
    'fin_auction_neighborhoods',
    'fin_auction_num_submitted_requests',
    'fin_auction_request_selection',
    'fin_auction_bundle_generation',
    'fin_auction_bundling_valuation',
    'fin_auction_num_auction_bundles',
    'fin_auction_bidding',
    'fin_auction_winner_determination',
    'fin_auction_num_auction_rounds'
]


