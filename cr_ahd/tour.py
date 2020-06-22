import vertex as vx
from utils import travel_time, opts


class Tour(object):
    def __init__(self, id_, sequence: list):
        self.id_ = id_
        self.sequence = sequence    # sequence of vertices
        # TODO: should this really be a sequence? What about the successor-list?
        self.arrival_schedule = [None] * len(sequence)  # arrival times
        self.service_schedule = [None] * len(sequence)  # start of service times
        self.cost = 0
        pass

    def __str__(self):
        sequence = [vertex.id_ for vertex in self.sequence]
        arrival_schedule = []
        service_schedule = []
        for i in range(len(self)):
            if self.arrival_schedule[i] is None:
                arrival_schedule.append(None)
                service_schedule.append(None)
            else:
                arrival_schedule.append(round(self.arrival_schedule[i], 2))
                service_schedule.append(round(self.service_schedule[i], 2))
        if self.cost is None:
            cost = None
        else:
            cost = round(self.cost)
        return f'ID:\t\t\t{self.id_}\nSequence:\t{sequence}\nArrival:\t{arrival_schedule}\nService:\t{service_schedule}\nCost:\t\t{cost}'

    def __len__(self):
        return len(self.sequence)

    def copy(self):
        tour = Tour(self.sequence)
        tour.arrival_schedule = self.arrival_schedule
        tour.service_schedule = self.service_schedule
        tour.cost = self.cost
        return tour

    # TODO: custom versions of the list methods, e.g. insert method that automatically updates the schedules?!
    def insert_and_reset_schedules(self, index, vertex: vx.Vertex, dist_matrix, ):
        self.sequence.insert(index, vertex)
        self.arrival_schedule = [None] * len(self)  # reset arrival times
        self.service_schedule = [None] * len(self)  # reset start of service times
        # self.compute_cost_and_schedules(dist_matrix)

        pass

    def insert_and_update_schedules(self):
        pass

    def compute_cost_and_schedules(self, dist_matrix, start_time=opts['start_time'], ignore_tw=True):
        self.cost = 0
        self.arrival_schedule[0] = start_time
        self.service_schedule[0] = start_time
        for i in range(1, len(self)):
            j: vx.Vertex = self.sequence[i - 1]
            k: vx.Vertex = self.sequence[i]
            dist = dist_matrix.loc[j.id_, k.id_]
            self.cost += dist
            planned_arrival = self.service_schedule[i - 1] + j.service_duration + travel_time(dist)
            if opts['verbose'] > 2:
                print(f'Planned arrival at {k}: {planned_arrival}')
            if not ignore_tw:
                assert planned_arrival <= k.tw.l
            self.arrival_schedule[i] = planned_arrival
            if planned_arrival >= k.tw.e:
                self.service_schedule[i] = planned_arrival
            else:
                self.service_schedule[i] = k.tw.e
        pass

    def is_feasible(self, dist_matrix, start_time=opts['start_time']):
        if opts['verbose'] > 1:
            print(f'== Fesibility Check')
            print(self)
        service_schedule = self.service_schedule.copy()
        service_schedule[0] = start_time
        for i in range(1, len(self)):
            j: vx.Vertex = self.sequence[i - 1]
            k: vx.Vertex = self.sequence[i]
            dist = dist_matrix.loc[j.id_, k.id_]
            if opts['verbose'] > 2:
                print(f'iteration {i}; service_schedule: {service_schedule}')
            earliest_arrival = service_schedule[i - 1] + j.service_duration + travel_time(dist)
            if earliest_arrival > k.tw.l:
                if opts['verbose'] > 0:
                    print(f'Infeasible! {round(earliest_arrival, 2)} > {k.id_}.tw.l: {k.tw.l}')
                return False
            elif earliest_arrival >= k.tw.e:
                service_schedule[i] = earliest_arrival
            else:
                service_schedule[i] = k.tw.e
        return True
