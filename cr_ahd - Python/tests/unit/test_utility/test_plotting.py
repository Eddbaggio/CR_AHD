import src.cr_ahd.utility_module.plotting as pl


# def test_plot_vertices(request_vertices_f):
#     pl.plot_vertices(request_vertices_f, True, True)

# def test_plot_vertices(request_vertices_c):
#     pl.plot_vertices(request_vertices_c, 'request_vertices_c', True, True)

# def test_plot_vertices(small_criss_cross_tour):
#     pl.plot_vertices(small_criss_cross_tour.routing_sequence, 'small_criss_cross_tour', True, True)

def test_plot_vertices(request_vertices_spiral):
    pl.plot_vertices(request_vertices_spiral(3, 3), 'Spiral Requests (not a tour)', True, True, True)


def test_plot_tour(spiral_tour):
    pl.plot_tour(spiral_tour(7), 'Spiral Tour', True, True, True, True)


def test_plot_carrier(carrier_spiral_partially_routed):
    pl.plot_carrier(carrier_spiral_partially_routed(5, 2, 14), 'Carrier with partially routed requests', show=True)
