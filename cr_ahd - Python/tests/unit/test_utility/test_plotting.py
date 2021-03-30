import src.cr_ahd.utility_module.plotting as pl


# def test_plot_vertices(request_vertices_f):
#     pl.plot_vertices(request_vertices_f, True, True)

def test_plot_vertices(request_vertices_c):
    pl.plot_vertices(request_vertices_c, 'request_vertices_c', True, True)
