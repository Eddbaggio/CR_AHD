import vertex as vx
from tour import Tour
from collections import OrderedDict
from utils import opts
from copy import deepcopy


class Carrier(object):
    """docstring for Carrier"""

    def __init__(self, id_: int, depot: vx.Vertex, vehicles: list):
        self.id_ = id_
        self.depot = depot
        self.vehicles = vehicles
        self.requests = OrderedDict()

        for v in vehicles:
            v.tour = Tour(id_=v.id_, sequence=[self.depot, self.depot])
        pass

    def __str__(self):
        return f'Carrier (ID:{self.id_}, Depot:{self.depot}, Vehicles:{len(self.vehicles)}, Requests:{len(self.requests)})'

    def assign_request(self, request: vx.Vertex):
        self.requests[request.id_] = request
        # TODO: add dist matrix attribute and extend it with each request assignment? -> inefficient, too much updating? instead have an extra function to extend the matrix whenever necessary?
        pass

    def _cheapest_insertion_construction(self, dist_matrix):
        """private method, call via l1_construction of carrier's instance instead"""
        unrouted: OrderedDict = self.requests.copy()
        tours = [v.tour for v in self.vehicles]
        # initialize tour with a seed, a) farthest un-routed request
        # or b) un-routed request with earliest deadline

        # here, b) is implemented
        seed = list(unrouted.values())[0]
        for key, request in unrouted.items():
            if request.tw.l < seed.tw.l:
                seed = unrouted[key]
        tour: Tour = tours[0]
        tour.insert_and_reset_schedules(index=1, vertex=seed, dist_matrix=dist_matrix)
        tour.compute_cost_and_schedules(dist_matrix=dist_matrix)
        unrouted.pop(seed.id_)

        def cheapest_feasible_insertion(u: vx.Vertex, tour: Tour):  # TODO: make this a method of the Tour class
            """calculating the cheapest insertion postition for an unrouted customer,
            returning cost and position"""
            if opts['verbose'] > 0:
                print(f'\n= Cheapest insertion of {u.id_} into {tour.id_}')
                print(tour)
            min_insertion_cost = 9999999
            i_star = None
            j_star = None

            # test all insertion positions
            for roh in range(1, len(tour)):
                i: vx.Vertex = tour.sequence[roh - 1]
                j: vx.Vertex = tour.sequence[roh]
                dist_i_u = dist_matrix.loc[i.id_, u.id_]
                dist_u_j = dist_matrix.loc[u.id_, j.id_]
                insertion_cost = dist_i_u + dist_u_j

                if opts['verbose'] > 1:
                    print(f'Between {i.id_} and {j.id_}: {insertion_cost}')

                if insertion_cost < min_insertion_cost:

                    # check feasibility
                    temp_tour = deepcopy(tour)
                    temp_tour.insert_and_reset_schedules(index=roh, vertex=u, dist_matrix=dist_matrix)
                    if temp_tour.is_feasible(dist_matrix=dist_matrix):
                        # TODO: there is a lot of potential to skip feasibility checks if the tw.l is smaller than the current arrival of its potential successor!

                        # update best known insertion position
                        min_insertion_cost = insertion_cost
                        i_star = i
                        j_star = j
                        insertion_position = roh
            if i_star:
                if opts['verbose'] > 0:
                    print(f'== Best: between {i_star.id_} and {j_star.id_}: {min_insertion_cost}')
                return insertion_position, min_insertion_cost
            else:
                raise IndexError('No feasible insertion position found')
            pass
        # def c1(u: vx.Vertex, alpha_1: float, mu):
        #     alpha_2 = 1 - alpha_1
        #     for roh in range(1, len(tour)):
        #         i: vx.Vertex = tour.sequence[roh - 1]
        #         j: vx.Vertex = tour.sequence[roh]

        #         c11 = dist_matrix.loc[i.id_, u.id_] + dist_matrix.loc[u.id_, j.id_] - mu * dist_matrix[i.id_, j.id_]
            # c12 =

            # c1 = alpha_1 * c11 + alpha_2 * c12

        while len(unrouted) > 0:
            _, u = unrouted.popitem(last=False)  # remove LIFO from unrouted
            inserted = False
            tour_index = 0
            while inserted is False:
                tour = tours[tour_index]
                try:
                    pos, cost = cheapest_feasible_insertion(u=u, tour=tour)
                    tour.insert_and_reset_schedules(pos, u, dist_matrix)  # add the unrouted element in its cheapest insertion position
                    tour.compute_cost_and_schedules(dist_matrix)
                    inserted = True
                except IndexError:
                    if opts['verbose'] > 0:
                        print(f'== Infeasible')
                    tour_index += 1
                    pass
        for v in self.vehicles:
            print()
            print(v.tour)
        pass
