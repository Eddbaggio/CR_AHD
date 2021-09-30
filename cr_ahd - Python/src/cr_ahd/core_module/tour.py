import datetime as dt
import logging.config
from copy import deepcopy
from typing import List, Sequence, Set, Dict

import utility_module.utils as ut

logger = logging.getLogger(__name__)


class Tour:
    def __init__(self, id_: int, depot_index: int):
        """

        :param id_: unique tour identifier
        :param depot_index: may be different to the id! id's can exist twice temporarily if a carrier is copied!
        """
        if isinstance(id_, int):
            logger.debug(f'Initializing tour {id_}')

        self.id_ = id_
        self.requests: Set[int] = set()  # collection of routed requests, in order of insertion! not in order of pickup

        # vertex data
        self.routing_sequence: List[int] = []  # vertices in order of service
        self.vertex_pos: Dict[int, int] = dict()  # mapping each vertex to its routing index
        self.arrival_time_sequence: List[dt.datetime] = []  # arrival time of each vertex
        self.service_time_sequence: List[dt.datetime] = []  # start of service time of each vertex
        self.wait_duration_sequence: List[dt.timedelta] = []  # required for efficient feasibility checks
        self.max_shift_sequence: List[dt.timedelta] = []  # required for efficient feasibility checks

        # sums
        self.sum_travel_distance: float = 0.0
        self.sum_travel_duration: dt.timedelta = dt.timedelta(0)
        self.sum_load: float = 0.0
        self.sum_revenue: float = 0.0
        self.sum_profit: float = 0.0

        # initialize depot to depot tour
        for _ in range(2):
            self.routing_sequence.insert(1, depot_index)
            self.arrival_time_sequence.insert(1, ut.EXECUTION_START_TIME)
            self.service_time_sequence.insert(1, ut.EXECUTION_START_TIME)
            self.wait_duration_sequence.insert(1, dt.timedelta(0))
            self.max_shift_sequence.insert(1, ut.END_TIME - ut.EXECUTION_START_TIME)

    def __str__(self):
        return f'Tour ID:\t{self.id_};\tRequests:\t{self.requests}\n' \
               f'Sequence:\t{self.routing_sequence}\n' \
               f'Arrival:\t{[x.strftime("%d-%H:%M:%S") for x in self.arrival_time_sequence]}\n' \
               f'Wait:\t\t{[str(x) for x in self.wait_duration_sequence]}\n' \
               f'Service:\t{[x.strftime("%d-%H:%M:%S") for x in self.service_time_sequence]}\n' \
               f'Max Shift:\t{[str(x) for x in self.max_shift_sequence]}\n' \
               f'Distance:\t{round(self.sum_travel_distance, 2)}\n' \
               f'Duration:\t{self.sum_travel_duration}\n' \
               f'Revenue:\t{self.sum_revenue}\n' \
               f'Profit:\t\t{self.sum_profit}\n'

    def __repr__(self):
        return f'Tour {self.id_} {self.requests}'

    def __len__(self):
        return len(self.routing_sequence)

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)

        setattr(result, 'id_', self.id_)
        setattr(result, 'requests', self.requests.copy())
        setattr(result, 'routing_sequence', self.routing_sequence[:])
        setattr(result, 'vertex_pos', self.vertex_pos.copy())
        setattr(result, 'arrival_time_sequence', self.arrival_time_sequence[:])
        setattr(result, 'service_time_sequence', self.service_time_sequence[:])
        setattr(result, 'wait_duration_sequence', self.wait_duration_sequence[:])
        setattr(result, 'max_shift_sequence', self.max_shift_sequence[:])
        setattr(result, 'sum_travel_distance', self.sum_travel_distance)
        setattr(result, 'sum_travel_duration', self.sum_travel_duration)
        setattr(result, 'sum_load', self.sum_load)
        setattr(result, 'sum_revenue', self.sum_revenue)
        setattr(result, 'sum_profit', self.sum_profit)

        return result

    @property
    def num_routing_stops(self):
        return len(self)

    def as_dict(self):
        return {
            'routing_sequence': self.routing_sequence,
            'arrival_schedule': self.arrival_time_sequence,
            'wait_sequence': self.wait_duration_sequence,
            'max_shift_sequence': self.max_shift_sequence,
            'service_schedule': self.service_time_sequence,
        }

    def print_as_table(self):
        print(f'Vertex\tArrival\t\tWait\t\tService\t\tMax_Shift')
        for v, a, w, s, m in zip(self.routing_sequence, self.arrival_time_sequence, self.wait_duration_sequence,
                                 self.service_time_sequence, self.max_shift_sequence):
            w_seconds = w.seconds
            w_hours = w_seconds // 3600
            w_seconds = w_seconds - (w_hours * 3600)
            w_minutes = w_seconds // 60
            w_seconds = w_seconds - (w_minutes * 60)

            m_seconds = m.seconds
            m_hours = m_seconds // 3600
            m_seconds = m_seconds - (m_hours * 3600)
            m_minutes = m_seconds // 60
            m_seconds = m_seconds - (m_minutes * 60)

            print(f'{v:02d}\t\t'
                  f'{a.strftime("%d-%H:%M:%S")}\t'
                  f'{w.days:02d}-{w_hours :02d}:{w_minutes :02d}:{w_seconds:02d}\t'
                  f'{s.strftime("%d-%H:%M:%S")}\t'
                  f'{m.days:02d}-{m_hours :02d}:{m_minutes :02d}:{m_seconds:02d}')

    def summary(self):
        return {
            # 'id_': self.id_,
            'sum_profit': self.sum_profit,
            'num_routing_stops': self.num_routing_stops,
            'sum_travel_distance': self.sum_travel_distance,
            'sum_travel_duration': self.sum_travel_duration,
            'sum_load': self.sum_load,
            'sum_revenue': self.sum_revenue,
        }

    def _single_insertion_feasibility_check(self, instance, insertion_index: int, insertion_vertex: int):
        """
        Checks whether the insertion of the insertion_vertex at insertion_pos is feasible.

        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local
        search for the team orienteering problem with time windows. Computers & Operations Research, 36(12),
        3281–3290. https://doi.org/10.1016/j.cor.2009.03.008

        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        :return: True if the insertion of the insertion_vertex at insertion_position is feasible, False otherwise
        """

        i = self.routing_sequence[insertion_index - 1]
        j = insertion_vertex
        k = self.routing_sequence[insertion_index]

        # [1] check precedence (only if the counterpart vertex is already in the tour)
        if instance.vertex_type(j) == 'delivery':
            pickup = j - instance.num_requests
            if pickup in self.vertex_pos and self.vertex_pos[pickup] > insertion_index:
                return False
        elif instance.vertex_type(j) == 'pickup':
            delivery = j + instance.num_requests
            if delivery in self.vertex_pos and self.vertex_pos[delivery] <= insertion_index:
                return False

        # [2] check max tour distance
        distance_shift_j = instance.distance([i, j], [j, k]) - instance.distance([i], [k])
        if self.sum_travel_distance + distance_shift_j > instance.vehicles_max_travel_distance:
            return False

        # [3] check time windows (NEW: in constant time!)
        # tw condition 1: start of service of j must fit the time window of j
        arrival_time_j = self.service_time_sequence[insertion_index - 1] + \
                         instance.vertex_service_duration[i] + \
                         instance.travel_duration([i], [j])
        tw_cond1 = arrival_time_j <= instance.tw_close[j]

        # tw condition 2: time_shift_j must be limited to the sum of wait_k + max_shift_k
        wait_j = max(dt.timedelta(0), instance.tw_open[j] - arrival_time_j)
        time_shift_j = instance.travel_duration([i], [j]) + \
                       wait_j + \
                       instance.vertex_service_duration[j] + \
                       instance.travel_duration([j], [k]) - \
                       instance.travel_duration([i], [k])
        wait_k = self.wait_duration_sequence[insertion_index]
        max_shift_k = self.max_shift_sequence[insertion_index]
        tw_cond2 = time_shift_j <= wait_k + max_shift_k

        if not tw_cond1 or not tw_cond2:
            return False

        # [4] check max vehicle load FIXME this is incorrect because it does not consider that load can increase and decrease during the tour!
        if self.sum_load + instance.vertex_load[j] > instance.vehicles_max_load:
            return False

        return True

    def insertion_feasibility_check(self,
                                    instance,
                                    insertion_indices: Sequence[int],
                                    insertion_vertices: Sequence[int]):
        """
        check whether an insertion of insertion_vertices at insertion_pos is feasible.

        :return: True if the combined insertion of all vertices in their corresponding positions is feasible, False
        otherwise
        """
        if len(insertion_indices) == 1:
            return self._single_insertion_feasibility_check(instance, insertion_indices[0], insertion_vertices[0])
        else:
            # sanity check whether insertion positions are sorted in ascending order
            assert all(insertion_indices[i] < insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))

            copy = deepcopy(self)

            # check all insertions sequentially
            for idx, (pos, vertex) in enumerate(zip(insertion_indices, insertion_vertices)):
                if copy._single_insertion_feasibility_check(instance, pos, vertex):
                    if idx < len(insertion_indices) - 1:  # to skip the last temporary insertion
                        copy._single_insert_and_update(instance, pos, vertex)
                else:
                    return False
            return True

    def _single_insert_and_update(self, instance, insertion_index: int, insertion_vertex: int):
        """
        ASSUMES THAT THE INSERTION WAS FEASIBLE, NO MORE CHECKS ARE EXECUTED IN HERE!

        insert a in a specified position of a routing sequence and update all related sequences, sums and schedules

        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
        orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
        https://doi.org/10.1016/j.cor.2009.03.008
        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        :return: the updated input variables (sequences, sums, schedules, ...) as a dict
        """

        assert 0 < insertion_index < len(self)
        assert 0 <= insertion_vertex < instance.num_carriers + instance.num_requests * 2

        # ===== [1] INSERT =====
        self.routing_sequence.insert(insertion_index, insertion_vertex)
        self.vertex_pos[insertion_vertex] = insertion_index

        i_index, i_vertex = insertion_index - 1, self.routing_sequence[insertion_index - 1]
        j_index, j_vertex = insertion_index, insertion_vertex
        k_index, k_vertex = insertion_index + 1, self.routing_sequence[insertion_index + 1]

        # calculate arrival at j_vertex (cannot use the _service_time_dict because of the depot)
        arrival_j = self.service_time_sequence[i_index] + \
                    instance.vertex_service_duration[i_vertex] + \
                    instance.travel_duration([i_vertex], [j_vertex])
        self.arrival_time_sequence.insert(insertion_index, arrival_j)

        # calculate start of service at j_vertex
        service_j = max(instance.tw_open[j_vertex], arrival_j)
        self.service_time_sequence.insert(insertion_index, service_j)

        # calculate wait duration at j_vertex
        wait_j = max(dt.timedelta(0), instance.tw_open[j_vertex] - arrival_j)
        self.wait_duration_sequence.insert(insertion_index, wait_j)

        # set max_shift of j_vertex temporarily to 0, will be updated further down
        max_shift_j = dt.timedelta(0)
        self.max_shift_sequence.insert(insertion_index, max_shift_j)

        # ===== [2] UPDATE =====
        # dist_shift: total distance consumption of inserting j_vertex in between i_vertex and k_vertex
        dist_shift_j = instance.distance([i_vertex], [j_vertex]) + \
                       instance.distance([j_vertex], [k_vertex]) - \
                       instance.distance([i_vertex], [k_vertex])

        # time_shift: total time consumption of inserting j_vertex in between i_vertex and k_vertex
        travel_time_shift_j = instance.travel_duration([i_vertex], [j_vertex]) + \
                              instance.travel_duration([j_vertex], [k_vertex]) - \
                              instance.travel_duration([i_vertex], [k_vertex])
        time_shift_j = travel_time_shift_j + \
                       wait_j + \
                       instance.vertex_service_duration[j_vertex]

        # update sums
        self.sum_travel_distance += dist_shift_j
        self.sum_travel_duration += travel_time_shift_j
        self.sum_load += instance.vertex_load[j_vertex]
        self.sum_revenue += instance.vertex_revenue[j_vertex]
        self.sum_profit = self.sum_profit + instance.vertex_revenue[j_vertex] - dist_shift_j

        # update arrival at k_vertex
        arrival_k = self.arrival_time_sequence[k_index] + time_shift_j
        self.arrival_time_sequence[k_index] = arrival_k

        # time_shift_k: how much of j_vertex's time shift is still available after waiting at k_vertex
        time_shift_k = max(dt.timedelta(0), time_shift_j - self.wait_duration_sequence[k_index])

        # update waiting time at k_vertex
        wait_k = max(dt.timedelta(0), self.wait_duration_sequence[k_index] - time_shift_j)
        self.wait_duration_sequence[k_index] = wait_k

        # update start of service at k_vertex
        service_k = self.service_time_sequence[k_index] + time_shift_k
        self.service_time_sequence[k_index] = service_k

        # update max shift of k_vertex
        max_shift_k = self.max_shift_sequence[k_index] - time_shift_k
        self.max_shift_sequence[k_index] = max_shift_k

        # increase vertex position record by 1 for all vertices succeeding j_vertex
        for vertex in self.routing_sequence[insertion_index + 1: -1]:
            self.vertex_pos[vertex] += 1

        # update data for all visits AFTER j_vertex until (a) shift == 0 or (b) the end is reached
        while time_shift_k > dt.timedelta(0) and k_index + 1 < len(self.routing_sequence):
            # move one forward
            k_index += 1
            k_vertex = self.routing_sequence[k_index]
            time_shift_j = time_shift_k

            # update arrival at k_vertex
            arrival_k = self.arrival_time_sequence[k_index] + time_shift_j
            self.arrival_time_sequence[k_index] = arrival_k

            time_shift_k = max(dt.timedelta(0), time_shift_j - self.wait_duration_sequence[k_index])

            # update wait duration
            wait_k = max(dt.timedelta(0), self.wait_duration_sequence[k_index] - time_shift_j)
            self.wait_duration_sequence[k_index] = wait_k

            # update service start time of k_vertex
            service_k = self.service_time_sequence[k_index] + time_shift_k
            self.service_time_sequence[k_index] = service_k

            # update max_shift of k_vertex
            max_shift_k = self.max_shift_sequence[k_index] - time_shift_k
            self.max_shift_sequence[k_index] = max_shift_k

        # update max_shift for visit j_vertex and visits PRECEDING the inserted vertex j_vertex
        for index in range(insertion_index, -1, -1):
            vertex = self.routing_sequence[index]

            max_shift_j = min(instance.tw_close[vertex] - self.service_time_sequence[index],
                              self.wait_duration_sequence[index + 1] + self.max_shift_sequence[index + 1])
            self.max_shift_sequence[index] = max_shift_j
        pass

    def insert_and_update(self, instance, insertion_indices: Sequence[int], insertion_vertices: Sequence[int]):
        """
        Inserts insertion_vertices at insertion_indices & updates the necessary data, e.g., arrival times.
        """
        assert all(insertion_indices[i] < insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))
        if isinstance(self.id_, int):
            logger.debug(f'Tour {self.id_}: Inserting {insertion_vertices} at {insertion_indices}')

        # execute all insertions sequentially:
        for index, vertex in zip(insertion_indices, insertion_vertices):
            self._single_insert_and_update(instance, index, vertex)

    # TODO
    # def insert_request_and_update(self, instance, request:int, pickup_pos:int, delivery_pos:int):
    #     self.requests.add(request)
    #     pass

    def _single_pop_and_update(self, instance, pop_index: int):
        """
        Following
        [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
        orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
        https://doi.org/10.1016/j.cor.2009.03.008
        [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
        delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
        https://doi.org/10.1016/j.ejor.2005.05.012

        :return: the popped vertex j at index pop_pos of the routing_sequence, as well as a dictionary of updated sums
        """

        # ===== [1] POP =====
        popped = self.routing_sequence.pop(pop_index)
        self.vertex_pos.pop(popped)

        index = pop_index
        i_vertex = self.routing_sequence[index - 1]
        j_vertex = popped
        k_vertex = self.routing_sequence[index]  # k_vertex has taken the place of j_vertex after j_vertex was removed

        self.arrival_time_sequence.pop(pop_index)

        self.service_time_sequence.pop(pop_index)

        wait_j = self.wait_duration_sequence.pop(pop_index)

        self.max_shift_sequence.pop(pop_index)

        # ===== [2] UPDATE =====

        # dist_shift: total distance reduction of removing j_vertex from in between i_vertex and k_vertex
        dist_shift_j = instance.distance([i_vertex], [k_vertex]) - \
                       instance.distance([i_vertex], [j_vertex]) - \
                       instance.distance([j_vertex], [k_vertex])

        # time_shift: total time reduction of removing j_vertex from in between i_vertex and k_vertex
        travel_time_shift_j = instance.travel_duration([i_vertex], [k_vertex]) - \
                              instance.travel_duration([i_vertex], [j_vertex]) - \
                              instance.travel_duration([j_vertex], [k_vertex])
        time_shift_j = travel_time_shift_j - wait_j - instance.vertex_service_duration[j_vertex]

        # update sums
        self.sum_travel_distance += dist_shift_j  # += since dist_shift_j will be negative
        self.sum_travel_duration += ut.travel_time(dist_shift_j)  # += since time_shift_j will be negative
        self.sum_load -= instance.vertex_load[j_vertex]
        self.sum_revenue -= instance.vertex_revenue[j_vertex]
        self.sum_profit = self.sum_profit - instance.vertex_revenue[j_vertex] - dist_shift_j

        # update the arrival at k_vertex
        arrival_k = self.arrival_time_sequence[index] + time_shift_j
        self.arrival_time_sequence[index] = arrival_k

        # update waiting time at k_vertex (more complicated than in insert) - can only increase
        wait_k = max(dt.timedelta(0), instance.tw_open[k_vertex] - self.arrival_time_sequence[index])
        self.wait_duration_sequence[index] = wait_k

        # time_shift_k: how much of i_vertex's time shift is still available after waiting at k_vertex?
        time_shift_k = min(dt.timedelta(0), time_shift_j + self.wait_duration_sequence[index])

        # update start of service at k_vertex
        service_k = max(instance.tw_open[k_vertex], self.arrival_time_sequence[index])
        self.service_time_sequence[index] = service_k

        # update max shift of k_vertex
        max_shift_k = self.max_shift_sequence[index] - time_shift_k
        self.max_shift_sequence[index] = max_shift_k

        # decrease vertex position record by 1 for all vertices succeeding j_vertex
        for vertex in self.routing_sequence[pop_index: -1]:
            self.vertex_pos[vertex] -= 1

        # update data for all visits AFTER j_vertex until (a) shift == 0 or (b) the end is reached
        while time_shift_k < dt.timedelta(0) and index + 1 < len(self.routing_sequence):
            # move one forward
            index += 1
            k_vertex = self.routing_sequence[index]
            time_shift_j = time_shift_k

            # update arrival at k_vertex
            arrival_k = self.arrival_time_sequence[index] + time_shift_j
            self.arrival_time_sequence[index] = arrival_k

            # update wait time at k_vertex
            wait_k = max(dt.timedelta(0), instance.tw_open[k_vertex] - self.arrival_time_sequence[index])
            self.wait_duration_sequence[index] = wait_k

            time_shift_k = min(dt.timedelta(0), time_shift_j + self.wait_duration_sequence[index])

            # service start time of k_vertex
            service_k = max(instance.tw_open[k_vertex], self.arrival_time_sequence[index])
            self.service_time_sequence[index] = service_k

            # update max_shift of k_vertex
            max_shift_k = self.max_shift_sequence[index] - time_shift_k
            self.max_shift_sequence[index] = max_shift_k

        # update max_shift for visits PRECEDING the removed vertex j_vertex
        for index in range(pop_index - 1, -1, -1):
            vertex = self.routing_sequence[index]
            max_shift_i = min(instance.tw_close[vertex] - self.service_time_sequence[index],
                              self.wait_duration_sequence[index + 1] + self.max_shift_sequence[index + 1])
            self.max_shift_sequence[index] = max_shift_i

        return popped

    # TODO
    # def pop_request_and_update(self, instance,  request: int):
    #     self.requests.remove(request)
    #     pickup, delivery = instance.pickup_and_delivery_pair(request)
    #     self._single_pop_and_update(instance, self.vertex_pos[pickup])
    #     self._single_pop_and_update(instance, self.vertex_pos[delivery])
    #     pass

    def pop_and_update(self, instance, pop_indices: Sequence[int]):

        """
        Removes vertices located at pop_indices
        :return: a list of popped vertices
        """

        # assure that indices are sorted
        assert all(pop_indices[i] <= pop_indices[i + 1] for i in range(len(pop_indices) - 1))

        popped = []

        # traverse the indices backwards to ensure that the succeeding indices are still correct once preceding ones
        # have been removed
        for pop_index in reversed(pop_indices):
            popped_vertex = self._single_pop_and_update(instance, pop_index)
            popped.append(popped_vertex)

        # reverse the popped array again to return vertices in the expected order
        return list(reversed(popped))

    def pop_distance_delta(self, instance, pop_indices: Sequence[int]):
        """
        :return: the negative delta that is obtained by popping the pop_indices from the routing sequence. NOTE: does
        not actually remove/pop the vertices
        """

        delta = 0

        # easy for single insertion
        if len(pop_indices) == 1:

            j_pos = pop_indices[0]
            i_vertex = self.routing_sequence[j_pos - 1]
            j_vertex = self.routing_sequence[j_pos]
            k_vertex = self.routing_sequence[j_pos + 1]

            delta += instance.distance([i_vertex], [k_vertex])
            delta -= instance.distance([i_vertex, j_vertex], [j_vertex, k_vertex])

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence.
        else:
            assert all(pop_indices[i] < pop_indices[i + 1] for i in
                       range(len(pop_indices) - 1)), f'Pop indices {pop_indices} are not in correct order'

            tmp_routing_sequence = list(self.routing_sequence)

            for j_pos in reversed(pop_indices):
                i_vertex = tmp_routing_sequence[j_pos - 1]
                j_vertex = tmp_routing_sequence.pop(j_pos)
                k_vertex = tmp_routing_sequence[j_pos]

                delta += instance.distance([i_vertex], [k_vertex])
                delta -= instance.distance([i_vertex, j_vertex], [j_vertex, k_vertex])

        return delta

    def insert_distance_delta(self, instance, insertion_indices: List[int], vertices: List[int]):
        """
        returns the distance surplus that is obtained by inserting the insertion_vertices at the insertion_positions.
        NOTE: Does not perform a feasibility check and does not actually insert the vertices!

        """
        delta = 0

        # easy for single insertion
        if len(insertion_indices) == 1:

            j_pos = insertion_indices[0]
            i_vertex = self.routing_sequence[j_pos - 1]
            j_vertex = vertices[0]
            k_vertex = self.routing_sequence[j_pos]

            delta += instance.distance([i_vertex, j_vertex], [j_vertex, k_vertex])
            delta -= instance.distance([i_vertex], [k_vertex])

        # must ensure that no edges are counted twice. Naive implementation with a tmp_routing_sequence
        else:
            assert all(insertion_indices[i] < insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))

            tmp_routing_sequence = list(self.routing_sequence)

            for j_pos, j_vertex in zip(insertion_indices, vertices):
                tmp_routing_sequence.insert(j_pos, j_vertex)

                i_vertex = tmp_routing_sequence[j_pos - 1]
                k_vertex = tmp_routing_sequence[j_pos + 1]

                delta += instance.distance([i_vertex, j_vertex], [j_vertex, k_vertex])
                delta -= instance.distance([i_vertex], [k_vertex])

        return delta

    def _single_insert_max_shift_delta(self, instance, insertion_index: int, insertion_vertex: int):
        """
        returns the change in max_shift time that would be observed if insertion_vertex was placed at
        insertion_index
        """

        # [1] compute wait_j and max_shift_j
        predecessor = self.routing_sequence[insertion_index - 1]
        successor = self.routing_sequence[insertion_index]

        arrival_j = max(self.arrival_time_sequence[insertion_index - 1], instance.tw_open[predecessor]) + \
                    instance.vertex_service_duration[predecessor] + \
                    instance.travel_duration([predecessor], [insertion_vertex])
        wait_j = max(dt.timedelta(0), instance.tw_open[insertion_vertex] - arrival_j)
        delta_ = instance.travel_duration([predecessor], [insertion_vertex]) + \
                 instance.travel_duration([insertion_vertex], [successor]) - \
                 instance.travel_duration([predecessor], [successor]) + \
                 instance.vertex_service_duration[insertion_vertex] + \
                 wait_j
        max_shift_j = min(instance.tw_close[insertion_vertex] - max(arrival_j, instance.tw_open[insertion_vertex]),
                          self.wait_duration_sequence[insertion_index] +
                          self.max_shift_sequence[insertion_index] -
                          delta_)

        # [2] algorithm 4.2 for max_shift delta of PRECEDING visits:
        # Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving
        # the pickup and delivery problem with time windows. European Journal of Operational Research, 175(2),
        # 672–687. https://doi.org/10.1016/j.ejor.2005.05.012

        beta = wait_j + max_shift_j
        predecessors_max_shift_delta = dt.timedelta(0)
        index = insertion_index - 1

        while True:
            if beta >= self.max_shift_sequence[index] or index == 0:
                break
            if index == insertion_index - 1:
                predecessors_max_shift_delta = self.max_shift_sequence[index] - beta
            elif self.wait_duration_sequence[index + 1] > dt.timedelta(0):
                predecessors_max_shift_delta += min(self.max_shift_sequence[index] - beta,
                                                    self.wait_duration_sequence[index + 1])
            beta += self.wait_duration_sequence[index]
            index -= 1

        # [3] delta in max_shift of insertion_vertex itself
        vertex_max_shift_delta = instance.tw_close[insertion_vertex] - \
                                 max(arrival_j, instance.tw_open[insertion_vertex]) - \
                                 max_shift_j

        # [4] delta in max_shift of succeeding vertices, which is exactly the travel time delta
        successors_max_shift_delta = instance.travel_duration([predecessor], [insertion_vertex]) + \
                                     instance.travel_duration([insertion_vertex], [successor]) - \
                                     instance.travel_duration([predecessor], [successor])

        return predecessors_max_shift_delta + vertex_max_shift_delta + successors_max_shift_delta

    def insert_max_shift_delta(self, instance, insertion_indices: List[int], insertion_vertices: List[int]):
        """
        returns the waiting time and max_shift time that would be assigned to insertion_vertices if they were
        inserted before insertion_indices
        """
        if len(insertion_indices) == 1:
            return self._single_insert_max_shift_delta(instance, insertion_indices[0], insertion_vertices[0])

        else:
            # sanity check whether insertion positions are sorted in ascending order
            assert all(insertion_indices[i] < insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))

            # create a temporary copy
            copy = deepcopy(self)

            # check all insertions sequentially
            total_max_shift_delta = dt.timedelta(0)
            for idx, (insertion_index, insertion_vertex) in enumerate(zip(insertion_indices, insertion_vertices)):
                max_shift_delta = copy._single_insert_max_shift_delta(instance, insertion_index, insertion_vertex)
                total_max_shift_delta += max_shift_delta
                copy._single_insert_and_update(instance, insertion_index, insertion_vertex)
            return total_max_shift_delta


# =====================================================================================================================
# stand-alone functions that are independent from the instance and solution classes but accept the raw data instead
# =====================================================================================================================


def single_insertion_feasibility_check(routing_sequence: Sequence[int],
                                       sum_travel_distance: float,
                                       service_schedule: Sequence[dt.datetime],
                                       sum_load: float,
                                       wait_sequence: Sequence[dt.timedelta],
                                       max_shift_sequence: Sequence[dt.timedelta],
                                       num_depots: int,
                                       num_requests: int,
                                       distance_matrix: Sequence[Sequence[float]],
                                       vertex_load: Sequence[float],
                                       service_duration: Sequence[dt.timedelta],
                                       vehicles_max_travel_distance: float,
                                       vehicles_max_load: float,
                                       tw_open: Sequence[dt.datetime],
                                       tw_close: Sequence[dt.datetime],
                                       insertion_index: int,
                                       insertion_vertex: int,
                                       ):
    """
    Checks whether the insertion of the insertion_vertex at insertion_pos is feasible.

    Following
    [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
    orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
    https://doi.org/10.1016/j.cor.2009.03.008
    [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
    delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
    https://doi.org/10.1016/j.ejor.2005.05.012

    :return: True if the insertion of the insertion_vertex at insertion_position is feasible, False otherwise
    """

    i = routing_sequence[insertion_index - 1]
    j = insertion_vertex
    k = routing_sequence[insertion_index]

    # check precedence if vertex is a delivery vertex
    if num_depots + num_requests <= j < num_depots + 2 * num_requests:
        precedence_vertex = j - num_requests
        if precedence_vertex not in routing_sequence[:insertion_index]:
            return False

    # check max tour distance
    distance_shift_j = distance_matrix[i][j] + distance_matrix[j][k] - distance_matrix[i][k]
    if sum_travel_distance + distance_shift_j > vehicles_max_travel_distance:
        return False

    # check time windows (NEW: in constant time!)
    travel_time_i_j = ut.travel_time(distance_matrix[i][j])
    travel_time_j_k = ut.travel_time(distance_matrix[j][k])
    travel_time_i_k = ut.travel_time(distance_matrix[i][k])

    # tw condition 1: service of j must fit the time window of j
    arrival_time_j = service_schedule[insertion_index - 1] + service_duration[i] + travel_time_i_j
    tw_cond1 = arrival_time_j <= tw_close[j]

    # tw condition 2: time_shift_j must be limited to the sum of wait_k + max_shift_k
    wait_j = max(dt.timedelta(0), tw_open[j] - arrival_time_j)
    time_shift_j = travel_time_i_j + wait_j + service_duration[j] + travel_time_j_k - travel_time_i_k
    wait_k = wait_sequence[insertion_index]
    max_shift_k = max_shift_sequence[insertion_index]
    tw_cond2 = time_shift_j <= wait_k + max_shift_k

    if not tw_cond1 or not tw_cond2:
        return False

    # check max vehicle load
    if sum_load + vertex_load[j] > vehicles_max_load:
        return False

    return True


def multi_insertion_feasibility_check(routing_sequence: List[int],
                                      sum_travel_distance: float,
                                      sum_travel_duration: dt.timedelta,
                                      arrival_schedule: List[dt.datetime],
                                      service_schedule: List[dt.datetime],
                                      sum_load: float,
                                      sum_revenue: float,
                                      sum_profit: float,
                                      wait_sequence: List[dt.timedelta],
                                      max_shift_sequence: List[dt.timedelta],
                                      num_depots,
                                      num_requests,
                                      distance_matrix: Sequence[Sequence[float]],
                                      vertex_load: Sequence[float],
                                      revenue: Sequence[float],
                                      service_duration: Sequence[dt.timedelta],
                                      vehicles_max_travel_distance,
                                      vehicles_max_load,
                                      tw_open: Sequence[dt.datetime],
                                      tw_close: Sequence[dt.datetime],
                                      insertion_indices,
                                      insertion_vertices,
                                      ):
    # sanity check whether insertion positions are sorted in ascending order
    assert all(insertion_indices[i] < insertion_indices[i + 1] for i in range(len(insertion_indices) - 1))
    input_dict = dict(
        routing_sequence=list(routing_sequence[:]),  # copy!
        sum_travel_distance=sum_travel_distance,
        sum_travel_duration=sum_travel_duration,
        arrival_schedule=list(arrival_schedule[:]),  # copy!
        service_schedule=list(service_schedule[:]),  # copy!
        sum_load=sum_load,
        sum_revenue=sum_revenue,
        sum_profit=sum_profit,
        wait_sequence=list(wait_sequence[:]),  # copy!
        max_shift_sequence=list(max_shift_sequence[:]),  # copy!
        num_depots=num_depots,
        num_requests=num_requests,
        distance_matrix=distance_matrix,
        vertex_load=vertex_load,
        revenue=revenue,
        service_duration=service_duration,
        vehicles_max_travel_distance=vehicles_max_travel_distance,
        vehicles_max_load=vehicles_max_load,
        tw_open=tw_open,
        tw_close=tw_close,
        insertion_index=None,
        insertion_vertex=None,
    )

    # check all insertions sequentially
    for idx, (pos, vertex) in enumerate(zip(insertion_indices, insertion_vertices)):
        input_dict['insertion_index'] = pos
        input_dict['insertion_vertex'] = vertex

        if single_insertion_feasibility_check(**input_dict):

            # if there are more insertions to check, insert the current vertex at the current pos inside the copied
            # route sequence & update the input dict before checking the next insertion
            if idx < len(insertion_indices) - 1:
                updated = single_insert_and_update(**input_dict)
                input_dict.update(updated)

        else:
            return False

    return True


def single_insert_and_update(routing_sequence: List[int],
                             sum_travel_distance: float,
                             sum_travel_duration: dt.timedelta,
                             arrival_schedule: List[dt.datetime],
                             service_schedule: List[dt.datetime],
                             sum_load: float,
                             sum_revenue: float,
                             sum_profit: float,
                             wait_sequence: List[dt.timedelta],
                             max_shift_sequence: List[dt.timedelta],
                             distance_matrix: Sequence[Sequence[float]],
                             vertex_load: Sequence[float],
                             revenue: Sequence[float],
                             service_duration: Sequence[dt.timedelta],
                             tw_open: Sequence[dt.datetime],
                             tw_close: Sequence[dt.datetime],
                             insertion_index: int,
                             insertion_vertex: int,
                             **kwargs
                             ):
    """
    ASSUMES THAT THE INSERTION WAS FEASIBLE, NO MORE CHECKS ARE EXECUTED IN HERE!

    insert a in a specified position of a routing sequence and update all related sequences, sums and schedules

    Following
    [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
    orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
    https://doi.org/10.1016/j.cor.2009.03.008
    [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
    delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
    https://doi.org/10.1016/j.ejor.2005.05.012

    :return: the updated input variables (sequences, sums, schedules, ...) as a dict
    """

    # [1] INSERT
    routing_sequence.insert(insertion_index, insertion_vertex)

    pos = insertion_index
    i = routing_sequence[pos - 1]
    j = insertion_vertex
    k = routing_sequence[pos + 1]

    # calculate arrival, start of service and wait for j
    arrival_schedule.insert(pos,
                            service_schedule[pos - 1] + service_duration[i] + ut.travel_time(distance_matrix[i][j]))
    service_schedule.insert(pos, max(tw_open[j], arrival_schedule[pos]))
    wait_sequence.insert(pos, max(dt.timedelta(0), tw_open[j] - arrival_schedule[pos]))
    max_shift_sequence.insert(pos, dt.timedelta(0))  # will be updated further down

    # [2] UPDATE

    # dist_shift: total distance consumption of inserting j in between i and k
    dist_shift_j = distance_matrix[i][j] + distance_matrix[j][k] - distance_matrix[i][k]

    # time_shift: total time consumption of inserting j in between i and k
    time_shift_j = ut.travel_time(dist_shift_j) + wait_sequence[pos] + service_duration[j]

    # update totals
    sum_travel_distance += dist_shift_j
    sum_travel_duration += ut.travel_time(dist_shift_j)
    sum_load += vertex_load[j]
    sum_revenue += revenue[j]
    sum_profit += revenue[j] - dist_shift_j

    # update arrival at k
    arrival_schedule[pos + 1] = arrival_schedule[pos + 1] + time_shift_j

    # time_shift_k: how much of j's time shift is still available after waiting at k
    time_shift_k = max(dt.timedelta(0), time_shift_j - wait_sequence[pos + 1])

    # update waiting time at k
    wait_sequence[pos + 1] = max(dt.timedelta(0), wait_sequence[pos + 1] - time_shift_j)

    # update start of service at k
    service_schedule[pos + 1] = service_schedule[pos + 1] + time_shift_k

    # update max shift of k
    max_shift_sequence[pos + 1] = max_shift_sequence[pos + 1] - time_shift_k

    pos += 1

    # update arrival, service, wait, max_shift and shift for all visits after j until shift == 0 or the end is reached
    while time_shift_k > dt.timedelta(0) and pos + 1 < len(routing_sequence):
        time_shift_j = time_shift_k

        arrival_schedule[pos + 1] = arrival_schedule[pos + 1] + time_shift_j
        time_shift_k = max(dt.timedelta(0), time_shift_j - wait_sequence[pos + 1])
        wait_sequence[pos + 1] = max(dt.timedelta(0), wait_sequence[pos + 1] - time_shift_j)
        service_schedule[pos + 1] = service_schedule[pos + 1] + time_shift_k
        max_shift_sequence[pos + 1] = max_shift_sequence[pos + 1] - time_shift_k
        pos += 1

    # update max_shift for visit j and visits PRECEDING the inserted vertex j
    for pos in range(insertion_index, -1, -1):
        i = routing_sequence[pos]
        max_shift_sequence[pos] = min(tw_close[i] - service_schedule[pos],
                                      wait_sequence[pos + 1] + max_shift_sequence[pos + 1])

        # if for vertex i, the max_shift does not change the insertion has no impact on a visit before i
        # if max_shift_sequence[pos] == old_max_shift:
        #     break

    # return the updated sums, sequences and schedules
    return dict(
        routing_sequence=routing_sequence,
        sum_travel_distance=sum_travel_distance,
        sum_travel_duration=sum_travel_duration,
        arrival_schedule=arrival_schedule,
        service_schedule=service_schedule,
        sum_load=sum_load,
        sum_revenue=sum_revenue,
        sum_profit=sum_profit,
        wait_sequence=wait_sequence,
        max_shift_sequence=max_shift_sequence,
    )


def multi_insert_and_update(routing_sequence: List[int],
                            sum_travel_distance: float,
                            sum_travel_duration: dt.timedelta,
                            arrival_schedule: List[dt.datetime],
                            service_schedule: List[dt.datetime],
                            sum_load: float,
                            sum_revenue: float,
                            sum_profit: float,
                            wait_sequence: List[dt.timedelta],
                            max_shift_sequence: List[dt.timedelta],
                            distance_matrix: Sequence[Sequence[float]],
                            vertex_load: Sequence[float],
                            revenue: Sequence[float],
                            service_duration: Sequence[dt.timedelta],
                            tw_open: Sequence[dt.datetime],
                            tw_close: Sequence[dt.datetime],
                            insertion_indices: Sequence[int],
                            insertion_vertices: Sequence[int]
                            ):
    """
    insert multiple vertices in multiple positions. route is updated after each insertion!

    """

    for index, vertex in zip(insertion_indices, insertion_vertices):
        updated_sums = single_insert_and_update(routing_sequence=routing_sequence,
                                                sum_travel_distance=sum_travel_distance,
                                                sum_travel_duration=sum_travel_duration,
                                                arrival_schedule=arrival_schedule,
                                                service_schedule=service_schedule,
                                                sum_load=sum_load,
                                                sum_revenue=sum_revenue,
                                                sum_profit=sum_profit,
                                                wait_sequence=wait_sequence,
                                                max_shift_sequence=max_shift_sequence,
                                                distance_matrix=distance_matrix,
                                                vertex_load=vertex_load,
                                                revenue=revenue,
                                                service_duration=service_duration,
                                                tw_open=tw_open,
                                                tw_close=tw_close,
                                                insertion_index=index,
                                                insertion_vertex=vertex
                                                )

        # the sums are fundamentals/primitives and must be updated manually
        sum_travel_distance = updated_sums['sum_travel_distance']
        sum_travel_duration = updated_sums['sum_travel_duration']
        sum_load = updated_sums['sum_load']
        sum_revenue = updated_sums['sum_revenue']
        sum_profit = updated_sums['sum_profit']

    return updated_sums


def single_pop_and_update(routing_sequence: List[int],
                          sum_travel_distance: float,
                          sum_travel_duration: dt.timedelta,
                          arrival_schedule: List[dt.datetime],
                          service_schedule: List[dt.datetime],
                          sum_load: float,
                          sum_revenue: float,
                          sum_profit: float,
                          wait_sequence: List[dt.timedelta],
                          max_shift_sequence: List[dt.timedelta],
                          distance_matrix: Sequence[Sequence[float]],
                          vertex_load: Sequence[float],
                          revenue: Sequence[float],
                          service_duration: Sequence[dt.timedelta],
                          tw_open: Sequence[dt.datetime],
                          tw_close: Sequence[dt.datetime],
                          pop_pos: int):
    """
    Following
    [1] Vansteenwegen,P., Souffriau,W., Vanden Berghe,G., & van Oudheusden,D. (2009). Iterated local search for the team
    orienteering problem with time windows. Computers & Operations Research, 36(12), 3281–3290.
    https://doi.org/10.1016/j.cor.2009.03.008
    [2] Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving the pickup and
    delivery problem with time windows. European Journal of Operational Research, 175(2), 672–687.
    https://doi.org/10.1016/j.ejor.2005.05.012

    :return: the popped vertex j at index pop_pos of the routing_sequence, as well as a dictionary of updated sums
    """

    # [1] POP
    popped = routing_sequence.pop(pop_pos)

    pos = pop_pos
    i = routing_sequence[pos - 1]
    j = popped
    k = routing_sequence[pos]  # k has taken the place of j after j has been removed

    arrival_schedule.pop(pop_pos)
    service_schedule.pop(pop_pos)
    wait_j = wait_sequence.pop(pop_pos)
    max_shift_sequence.pop(pop_pos)

    # [2] UPDATE

    # dist_shift: total distance reduction of removing j from in between i and k
    dist_shift_j = -distance_matrix[i][j] - distance_matrix[j][k] + distance_matrix[i][k]  # negative value!

    # time_shift: total time reduction of removing j from in between i and k
    time_shift_j = ut.travel_time(dist_shift_j) - wait_j - service_duration[j]  # negative value!

    # update totals
    sum_travel_distance += dist_shift_j  # += since dist_shift_j will be negative
    sum_travel_duration += ut.travel_time(dist_shift_j)  # += since time_shift_j will be negative
    sum_load -= vertex_load[j]
    sum_revenue -= revenue[j]
    sum_profit = sum_profit - revenue[j] - dist_shift_j

    # update the arrival at k
    arrival_schedule[pos] = arrival_schedule[pos] + time_shift_j

    # update waiting time at k (more complicated than in insert) - can only increase
    wait_sequence[pos] = max(dt.timedelta(0), tw_open[k] - arrival_schedule[pos])

    # time_shift_k: how much of j's time shift is still available after waiting at k
    # of the time gained by removal of j, how much is still available after k?
    time_shift_k = min(dt.timedelta(0), time_shift_j + wait_sequence[pos])

    # update start of service at k
    service_schedule[pos] = max(tw_open[k], arrival_schedule[pos])

    # update max shift of k
    max_shift_sequence[pos] = max_shift_sequence[pos] - time_shift_k

    pos += 1

    # update arrival, service, wait, max_shift and shift for all visits after j until shift == 0
    while time_shift_k < dt.timedelta(0) and pos < len(routing_sequence):
        time_shift_j = time_shift_k
        k = routing_sequence[pos]

        arrival_schedule[pos] = arrival_schedule[pos] + time_shift_j
        wait_sequence[pos] = max(dt.timedelta(0), tw_open[k] - arrival_schedule[pos])
        time_shift_k = min(dt.timedelta(0), time_shift_j + wait_sequence[pos])
        service_schedule[pos] = max(tw_open[k], arrival_schedule[pos])
        max_shift_sequence[pos] = max_shift_sequence[pos] - time_shift_k
        pos += 1

    # update max_shift for visits PRECEDING the removed vertex j
    for pos in range(pop_pos - 1, -1, -1):
        i = routing_sequence[pos]
        max_shift_sequence[pos] = min(tw_close[i] - service_schedule[pos],
                                      wait_sequence[pos + 1] + max_shift_sequence[pos + 1])

        # if for vertex i, the max_shift does not change the insertion has no impact on a visit before i
        # if max_shift_sequence[pos] == old_max_shift:
        #     break

    return j, dict(
        routing_sequence=routing_sequence,
        sum_travel_distance=sum_travel_distance,
        sum_travel_duration=sum_travel_duration,
        arrival_schedule=arrival_schedule,
        service_schedule=service_schedule,
        sum_load=sum_load,
        sum_revenue=sum_revenue,
        sum_profit=sum_profit,
        wait_sequence=wait_sequence,
        max_shift_sequence=max_shift_sequence,
    )


def multi_pop_and_update(routing_sequence: List[int],
                         sum_travel_distance: float,
                         sum_travel_duration: dt.timedelta,
                         arrival_schedule: List[dt.datetime],
                         service_schedule: List[dt.datetime],
                         sum_load: float,
                         sum_revenue: float,
                         sum_profit: float,
                         wait_sequence: List[dt.timedelta],
                         max_shift_sequence: List[dt.timedelta],
                         distance_matrix: Sequence[Sequence[float]],
                         vertex_load: Sequence[float],
                         revenue: Sequence[float],
                         service_duration: Sequence[dt.timedelta],
                         tw_open: Sequence[dt.datetime],
                         tw_close: Sequence[dt.datetime],
                         pop_indices: Sequence[int]):
    """

    :return: a list of popped vertices as well as a dictionary of updated sums
    """

    # assure that indices are sorted
    assert all(pop_indices[i] <= pop_indices[i + 1] for i in range(len(pop_indices) - 1))

    popped = []

    # traverse the indices backwards to ensure that the succeeding indices are still correct once preceding ones
    # have been removed
    for pop_index in reversed(pop_indices):
        j, updated_sums = single_pop_and_update(routing_sequence=routing_sequence,
                                                sum_travel_distance=sum_travel_distance,
                                                sum_travel_duration=sum_travel_duration,
                                                arrival_schedule=arrival_schedule,
                                                service_schedule=service_schedule,
                                                sum_load=sum_load,
                                                sum_revenue=sum_revenue,
                                                sum_profit=sum_profit,
                                                wait_sequence=wait_sequence,
                                                max_shift_sequence=max_shift_sequence,
                                                distance_matrix=distance_matrix,
                                                vertex_load=vertex_load,
                                                revenue=revenue,
                                                service_duration=service_duration,
                                                tw_open=tw_open,
                                                tw_close=tw_close,
                                                pop_pos=pop_index)
        popped.append(j)
        sum_travel_distance = updated_sums['sum_travel_distance']
        sum_travel_duration = updated_sums['sum_travel_duration']
        sum_load = updated_sums['sum_load']
        sum_revenue = updated_sums['sum_revenue']
        sum_profit = updated_sums['sum_profit']

    # reverse the popped array again to return things in the expected order
    return list(reversed(popped)), updated_sums


def single_insert_max_shift_delta(routing_sequence: List[int],
                                  arrival_schedule: List[dt.datetime],
                                  wait_sequence: List[dt.timedelta],
                                  max_shift_sequence: List[dt.timedelta],
                                  distance_matrix: Sequence[Sequence[float]],
                                  service_duration: Sequence[dt.timedelta],
                                  tw_open: Sequence[dt.datetime],
                                  tw_close: Sequence[dt.datetime],
                                  insertion_index: int,
                                  insertion_vertex: int,
                                  **kwargs
                                  ):
    """returns the change in max_shift time that would be observed if insertion_vertex was placed before
    insertion_index """

    # [1] compute wait_j and max_shift_j
    predecessor = routing_sequence[insertion_index - 1]
    successor = routing_sequence[insertion_index]
    arrival_j = max(arrival_schedule[insertion_index - 1], tw_open[predecessor]) + service_duration[
        predecessor] + ut.travel_time(distance_matrix[predecessor][insertion_vertex])
    wait_j = max(dt.timedelta(0), tw_open[insertion_vertex] - arrival_j)
    delta_ = ut.travel_time(distance_matrix[predecessor][insertion_vertex]) + \
             ut.travel_time(distance_matrix[insertion_vertex][successor]) - \
             ut.travel_time(distance_matrix[predecessor][successor]) + \
             service_duration[insertion_vertex] + \
             wait_j
    max_shift_j = min(tw_close[insertion_vertex] - max(arrival_j, tw_open[insertion_vertex]),
                      wait_sequence[insertion_index] + max_shift_sequence[insertion_index] - delta_)

    # [2] algorithm 4.2 for max_shift delta of PRECEDING visits:
    # Lu,Q., & Dessouky,M.M. (2006). A new insertion-based construction heuristic for solving
    # the pickup and delivery problem with time windows. European Journal of Operational Research, 175(2),
    # 672–687. https://doi.org/10.1016/j.ejor.2005.05.012

    beta = wait_j + max_shift_j
    predecessors_max_shift_delta = dt.timedelta(0)
    k = insertion_index - 1

    while True:
        if beta >= max_shift_sequence[k] or k == 0:
            break

        if k == insertion_index - 1:
            predecessors_max_shift_delta = max_shift_sequence[k] - beta

        elif wait_sequence[k + 1] > dt.timedelta(0):
            predecessors_max_shift_delta += min(max_shift_sequence[k] - beta, wait_sequence[k + 1])

        beta += wait_sequence[k]
        k -= 1

    # [3] delta in max_shift of insertion_vertex itself
    vertex_max_shift_delta = tw_close[insertion_vertex] - max(arrival_j, tw_open[insertion_vertex]) - max_shift_j

    # [4] delta in max_shift of succeeding vertices, which is exactly the travel time delta
    successors_max_shift_delta = ut.travel_time(distance_matrix[predecessor][insertion_vertex] +
                                                distance_matrix[insertion_vertex][successor] -
                                                distance_matrix[predecessor][successor])

    return predecessors_max_shift_delta + vertex_max_shift_delta + successors_max_shift_delta  # c1 + c2 + c3


def multi_insert_max_shift_delta(routing_sequence: Sequence[int],
                                 sum_travel_distance: float,
                                 sum_travel_duration: dt.timedelta,
                                 arrival_schedule: Sequence[dt.datetime],
                                 service_schedule: Sequence[dt.datetime],
                                 sum_load: float,
                                 sum_revenue: float,
                                 sum_profit: float,
                                 wait_sequence: Sequence[dt.timedelta],
                                 max_shift_sequence: Sequence[dt.timedelta],
                                 distance_matrix: Sequence[Sequence[float]],
                                 vertex_load: Sequence[float],
                                 revenue: Sequence[float],
                                 service_duration: Sequence[dt.timedelta],
                                 tw_open: Sequence[dt.datetime],
                                 tw_close: Sequence[dt.datetime],
                                 insertion_indices: Sequence[int],
                                 insertion_vertices: Sequence[int]):
    """returns the waiting time and max_shift time that would be assigned to insertion_vertices if they were inserted
    before insertion_indices"""
    input_dict = dict(
        routing_sequence=list(routing_sequence[:]),  # copy!
        sum_travel_distance=sum_travel_distance,
        sum_travel_duration=sum_travel_duration,
        arrival_schedule=list(arrival_schedule[:]),  # copy!
        service_schedule=list(service_schedule[:]),  # copy!
        sum_load=sum_load,
        sum_revenue=sum_revenue,
        sum_profit=sum_profit,
        wait_sequence=list(wait_sequence[:]),  # copy!
        max_shift_sequence=list(max_shift_sequence[:]),  # copy!
        distance_matrix=distance_matrix,
        vertex_load=vertex_load,
        revenue=revenue,
        service_duration=service_duration,
        tw_open=tw_open,
        tw_close=tw_close,
        insertion_index=None,
        insertion_vertex=None,
    )

    total_max_shift_delta = dt.timedelta(0)
    for idx, (index, vertex) in enumerate(zip(insertion_indices, insertion_vertices)):
        input_dict['insertion_index'] = index
        input_dict['insertion_vertex'] = vertex
        max_shift_delta = single_insert_max_shift_delta(**input_dict)
        total_max_shift_delta += max_shift_delta

        # if there are more vertices that must be considered, insert the current vertex at the current pos
        if idx < len(insertion_indices) - 1:
            updated = single_insert_and_update(**input_dict)
            input_dict.update(updated)

    return total_max_shift_delta
