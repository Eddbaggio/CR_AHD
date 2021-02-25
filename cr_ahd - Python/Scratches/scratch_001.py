"""
Trying to figure out how I can ensure that any tour that is only temporary (e.g. for testing insertion costs etc.)
is destructed properly. In particular, all the request that were held by that tour must be set to vertex.routed = False.
Good Ideas:
a) use a context manager and the "with" statements (__enter__, __exit__)
Bad Ideas:
b) use a destructor (__del__) <-- unreliable, no control over when __del__ is called
c) create local copies of everything that is only required temporarily (tour, vertices, ...)
d) add temp_tour and temp_vertex classes

----------

actually, none of the above is perfect: They all require to have a deepcopy of the tour, that means the tour as well as
the vertices (with the same id!) exist twice for a while.
Better ? : test insertion in the *original* tour and ensure to remove the vertex afterwards.
    is there any reason to NOT use the original tour in I1 insertion?:

"""
from tour import Tour
import vertex as vx
from helper.utils import make_dist_matrix


def TwoOpt(tour: Tour):
    tour.compute_cost_and_schedules()
    best_cost = tour.cost
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue  # no effect
                # the actual 2opt swap
                tour.reverse_section(i, j)
                # check for improvement
                tour.compute_cost_and_schedules()
                print(tour.routing_sequence)
                if tour.cost < best_cost and tour.is_feasible():
                    improved = True
                    best_cost = tour.cost
                # if no improvement -> undo the 2opt swap
                else:
                    tour.reverse_section(i, j)


if __name__ == '__main__':
    depot = vx.DepotVertex('depot', 0, 0)
    requests = [vx.Vertex(f'{i}', i, i, 0, 0, 1000) for i in range(5)]
    distance_matrix = make_dist_matrix([*requests, depot])
    tour = Tour('tour', depot, distance_matrix)

    TwoOpt(tour)
