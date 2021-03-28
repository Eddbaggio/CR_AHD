import src.cr_ahd.auction_module.bundle_generation as bg
from src.cr_ahd.utility_module.utils import make_travel_dist_matrix, Coordinates, flatten_dict_of_lists


# def test_AllBundles(submitted_requests_01):
#     dist_matrix = make_dist_matrix(flatten_dict_of_lists(submitted_requests_01))
#     bundles = bg.AllBundles(dist_matrix).execute(submitted_requests_01)
#     assert

def test_random_bundle_set_generation(submitted_requests_random_15):
    """cannot actually be tested since it's random"""
    distance_matrix = make_travel_dist_matrix(flatten_dict_of_lists(submitted_requests_random_15))
    bundle_set = bg.RandomPartition(distance_matrix).execute(submitted_requests_random_15)
    assert len(bundle_set) <= len(submitted_requests_random_15)


def test_k_means_bundle_set_generation(submitted_requests_a):
    """"""
    distance_matrix = make_travel_dist_matrix(flatten_dict_of_lists(submitted_requests_a))
    bundle_set = bg.KMeansBundles(distance_matrix).execute(submitted_requests_a)
    bundle_0_ids = sorted([v.id_ for v in bundle_set[0]])
    assert bundle_0_ids == ['r0', 'r1', 'r2', 'r3']
    bundle_1_ids = sorted([v.id_ for v in bundle_set[1]])
    assert bundle_1_ids == ['r4', 'r5', 'r6', 'r7']


# GanstererProxyBundles
# def test_bundle_centroid(request_vertices_b, depot_vertex):
#     distance_matrix = make_dist_matrix([*request_vertices_b, depot_vertex])
#     centroid = bg.GanstererProxyBundles(distance_matrix).bundle_centroid(request_vertices_b, depot_vertex)
#     assert centroid.x == centroid.y == 5.954915028125262


# def test_bundle_radius(request_vertices_b, depot_vertex):
#     distance_matrix = make_dist_matrix([*request_vertices_b, depot_vertex])
#     radius = bg.GanstererProxyBundles(distance_matrix).bundle_radius(request_vertices_b, depot_vertex)
#     assert radius == 10.90742919249142


# def test_bundle_density(request_vertices_b, depot_vertex):
#     # TODO too many distance matrices, group tests into a TestClass!
#     distance_matrix = make_dist_matrix([*request_vertices_b, depot_vertex])
#     density = bg.GanstererProxyBundles(distance_matrix).bundle_density(request_vertices_b, depot_vertex)
#     assert density == 1.107030134733417


# def test_tour_length(request_vertices_b, depot_vertex):
#     distance_matrix = make_dist_matrix([*request_vertices_b, depot_vertex])
#     tour_length = bg.GanstererProxyBundles(distance_matrix).bundle_tour_length(request_vertices_b, depot_vertex)
#     assert tour_length == 62.42640687119285
