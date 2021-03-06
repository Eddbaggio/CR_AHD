import datetime as dt
import itertools
import math
import re
import sys
from collections import namedtuple
from typing import List, Sequence, Tuple

from matplotlib.colors import LinearSegmentedColormap
from tqdm import trange

from tw_management_module.time_window import TimeWindow

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


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def hamming_distance(string1: Sequence, string2: Sequence):
    """
    The Hamming distance between two equal-length strings of symbols is the number of positions at which the
    corresponding symbols are different.
    Straight up copied from wikipedia: https://en.wikipedia.org/wiki/Hamming_distance

    :return: number of positions at which string1 and string2 are different
    """
    assert len(string1) == len(string2)
    return sum(xi != yi for xi, yi in zip(string1, string2))


def flatten(sequence: (List, Tuple)):
    if not sequence:
        return sequence
    if isinstance(sequence[0], (List, Tuple)):
        return flatten(sequence[0]) + flatten(sequence[1:])
    return sequence[:1] + flatten(sequence[1:])


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
    pickup_x, pickup_y = instance.vertex_x_coords[pickup_vertex], instance.vertex_y_coords[delivery_vertex]
    delivery_x, delivery_y = instance.vertex_x_coords[pickup_vertex], instance.vertex_y_coords[delivery_vertex]
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


def datetime_range(start: dt.datetime, stop: dt.datetime, step: dt.timedelta = None, num: int = None, startpoint=True,
                   endpoint=True):
    """
    returns a generator of equally-spaced datetime objects. Exactly one of step or num must be supplied.
    """

    assert None in (step, num), f'only one of step or num must be given'
    if step is not None:
        assert step.total_seconds() > 0, f"Step shouldn't be 0"
    if num is not None:
        assert num > 0, f"Num shouldn't be 0"

    delta = stop - start

    if num:
        if endpoint and startpoint:
            div = num - 1
        elif endpoint or startpoint:
            div = num
        else:
            div = num + 1
        step = delta / div
    else:
        num = delta // step
        if endpoint and startpoint:
            num += 1
        elif endpoint or startpoint:
            pass
        else:
            num -= 1

    if not startpoint:
        start += step

    return (start + (x * step) for x in range(num))


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(str(s))]


def indices_to_nested_lists(indices: Sequence[int], elements: Sequence):
    nested_list = [[] for _ in range(max(indices) + 1)]
    for x, y in zip(elements, indices):
        nested_list[y].append(x)
    return nested_list


# TODO can this not be a method of the solution itself? (circular dependencies solution - instance?)
def validate_solution(instance, solution):
    assert solution.num_carriers() > 0

    for carrier_id in trange(len(solution.carriers), desc=f'Solution validation', disable=True):
        carrier = solution.carriers[carrier_id]
        assert len(
            carrier.unrouted_requests) == 0, f'Carrier {carrier} has unrouted requests: {carrier.unrouted_requests}'
        for tour in carrier.tours:
            assert tour is solution.tours[tour.id_]
            assert tour.routing_sequence[0] == carrier.id_, \
                f'the start of tour {tour} from carrier {carrier} is not a depot'
            assert tour.routing_sequence[-1] == carrier.id_, \
                f'the end of tour {tour} from carrier {carrier} is not a depot'

            validate_tour(instance, tour)

            # request-to-tour assignment record
            # for vertex in tour.routing_sequence[1:-1]:
            #     request = instance.request_from_vertex(vertex)
            #     assert solution.request_to_tour_assignment[
            #                request] == tour.id_, f'{instance.id_}, tour {tour.id_}, vertex {vertex} at index {i}'


# TODO can this not be a method of the solution itself? (circular dependencies tour - instance?)
def validate_tour(instance, tour):
    assert tour.sum_load == sum([instance.vertex_revenue[v] for v in tour.routing_sequence]), instance.id_
    assert tour.sum_travel_distance <= instance.max_tour_distance, instance.id_
    assert tour.sum_travel_duration <= instance.max_tour_duration, instance.id_
    # assert round(tour.sum_profit, 4) == round(tour.sum_revenue - tour.sum_travel_distance, 4), \
    #     f'{instance.id_}: {round(tour.sum_profit, 4)}!={round(tour.sum_revenue - tour.sum_travel_distance, 4)}'

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

        '''
        # precedence constraint
        if instance.vertex_type(vertex) == 'pickup':
            assert vertex + instance.num_requests in tour.routing_sequence[i:], msg
        elif instance.vertex_type(vertex) == 'delivery':
            assert vertex - instance.num_requests in tour.routing_sequence[:i], msg
        else:
            assert vertex in range(instance.num_carriers), msg
        '''

        # meta data
        if instance.vertex_type(vertex) != 'depot':
            assert tour.vertex_pos[vertex] == i, msg


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


ACCEPTANCE_START_TIME: dt.datetime = dt.datetime.min

EXECUTION_START_TIME: dt.datetime = ACCEPTANCE_START_TIME + dt.timedelta(days=1, hours=10)
END_TIME: dt.datetime = EXECUTION_START_TIME + dt.timedelta(hours=8)
EXECUTION_TIME_HORIZON = TimeWindow(EXECUTION_START_TIME, END_TIME)


def generate_time_windows(tw_length: dt.timedelta,
                          start: dt.datetime = EXECUTION_START_TIME,
                          end: dt.datetime = END_TIME):
    return [TimeWindow(e, min(e + tw_length, end)) for e in datetime_range(start, end, step=tw_length, endpoint=False)]
