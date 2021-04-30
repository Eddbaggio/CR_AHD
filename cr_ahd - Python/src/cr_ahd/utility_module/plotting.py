from typing import Union, List
import pandas as pd
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import plotly.express as px
import plotly.graph_objects as go
from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut

config = dict({'scrollZoom': True})


def _vertex_figure(instance: it.PDPInstance, solution: slt.GlobalSolution,
                   vertices: List[int], title: str = 'Scatter Plot - Vertices'):
    df = pd.DataFrame({
        'id_': vertices,
        'x': [instance.x_coords[v] for v in vertices],
        'y': [instance.y_coords[v] for v in vertices],
        'tw_open': [solution.tw_open[v] for v in vertices],
        'tw_close': [solution.tw_close[v] for v in vertices],
    })
    fig = px.scatter(
        data_frame=df,
        x='x',
        y='y',
        text='id_',
        hover_name='id_',
        hover_data=['x', 'y', 'tw_open', 'tw_close'],
        size=[2.5] * len(vertices),
        title=title
    )
    return fig


def _add_vertex_annotations(fig: go.Figure, instance: it.PDPInstance, solution: slt.GlobalSolution,
                            vertices: List[int]):
    """creates annotations for the time window of each vertex"""
    for v in vertices:
        time_format = "Day %d %H:%M:%S"
        fig.add_annotation(
            x=instance.x_coords[v], y=instance.y_coords[v],
            text=f'[{solution.tw_open[v].strftime(time_format)} - {solution.tw_close[v].strftime(time_format)}]',
            clicktoshow='onoff',
            yshift=-25, showarrow=False)
    pass


def _add_tour_annotations(fig: go.Figure, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int,
                          tour: int, arrival_time=True, service_time=True):
    t = solution.carrier_solutions[carrier].tours[tour]
    for index, vertex in enumerate(t.routing_sequence):
        time_format = "Day %d %H:%M:%S"
        text = ''
        if arrival_time:
            text += f'Arrival: {t.arrival_schedule[index].strftime(time_format)}'
        if service_time:
            text += f'<br>Service: {t.service_schedule[index].strftime(time_format)}'
            fig.add_annotation(x=instance.x_coords[vertex], y=instance.y_coords[vertex],
                               text=text,
                               clicktoshow='onoff',
                               yshift=-50, showarrow=False)
    pass


def _add_edge_traces(fig: go.Figure, instance: it.PDPInstance, vertices: List[int]):
    for i in range(len(vertices) - 1):
        arrow_tail = (instance.x_coords[vertices[i]], instance.y_coords[vertices[i]])
        arrow_head = (instance.x_coords[vertices[i + 1]], instance.y_coords[vertices[i + 1]])
        fig.add_annotation(x=arrow_head[0],
                           y=arrow_head[1],
                           ax=arrow_tail[0],
                           ay=arrow_tail[1],
                           xref='x',
                           yref='y',
                           axref='x',
                           ayref='y',
                           text="",
                           showarrow=True,
                           arrowhead=2,
                           arrowsize=2,
                           arrowwidth=1,
                           arrowcolor='black',
                           standoff=10,
                           startstandoff=10)
    pass


def plot_vertices(instance, solution, vertices: List[int], title: str, annotations=False, edges: bool = False,
                  show: bool = False):
    fig = _vertex_figure(instance, solution, vertices, title)
    if annotations:
        _add_vertex_annotations(fig, instance, solution, vertices)
    if edges:
        _add_edge_traces(fig, instance, vertices)
    if show:
        fig.show(config=config)
    return fig


def plot_tour(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier, tour, title,
              time_windows: bool = True, arrival_times: bool = True, service_times: bool = True, show: bool = False):
    t = solution.carrier_solutions[carrier].tours[tour]
    fig = _vertex_figure(instance, solution, t.routing_sequence, title)
    _add_edge_traces(fig, instance, t.routing_sequence)
    if time_windows:
        _add_vertex_annotations(fig, instance, solution, t.routing_sequence)
    if arrival_times or service_times:
        _add_tour_annotations(fig, instance, solution, carrier, tour, arrival_times, service_times)
    if show:
        fig.show(config=config)
    return fig


def plot_carrier(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier, title='', tours: bool = True,
                 time_windows: bool = True, arrival_times: bool = True, service_times: bool = True, show: bool = False):
    vertices = [v for t in solution.carrier_solutions[carrier].tours for v in t.routing_sequence]
    fig = _vertex_figure(instance, solution, vertices, title)
    if tours:
        for tour in solution.carrier_solutions[carrier].tours:
            _add_edge_traces(fig, instance, tour.routing_sequence)
    if time_windows:
        _add_vertex_annotations(fig, instance, solution, vertices)
    if arrival_times or service_times:
        for tour in solution.carrier_solutions[carrier].tours:
            _add_tour_annotations(fig, instance, solution, carrier, tour)
    if show:
        fig.show(config=config)


def _make_tour_scatter(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
    tour_ = solution.carrier_solutions[carrier].tours[tour]

    df = pd.DataFrame({
        'id_': tour_.routing_sequence[:-1],
        'x': [instance.x_coords[v] for v in tour_.routing_sequence[:-1]],
        'y': [instance.y_coords[v] for v in tour_.routing_sequence[:-1]],
        'revenue': [instance.revenue[v] for v in tour_.routing_sequence[:-1]],
        'tw_open': [solution.tw_open[v] for v in tour_.routing_sequence[:-1]],
        'tw_close': [solution.tw_close[v] for v in tour_.routing_sequence[:-1]], })
    original_carrier_assignment = [instance.request_to_carrier_assignment[r] for r in
                                   [instance.request_from_vertex(v) for v in tour_.routing_sequence[1:-1]]]
    original_carrier_assignment.insert(0, carrier)
    hover_text = []
    for v in tour_.routing_sequence[1:-1]:
        hover_text.append(
            f"Request {instance.request_from_vertex(v)}</br>{instance.vertex_type(v)}</br>"
            f"Revenue: {instance.revenue[v]}</br>"
            f"Tour's Travel Distance: {tour_.sum_travel_distance}")
    hover_text.insert(0, f'Depot {carrier}')

    # colorscale = [(c, ut.univie_colors_100[c]) for c in range(instance.num_carriers)]
    return go.Scatter(x=df['x'], y=df['y'], mode='markers+text',
                      marker=dict(
                          symbol=['square', *['circle'] * (tour_.num_routing_stops - 2)],
                          size=ut.linear_interpolation(df['revenue'].values, 15, 30),
                          line=dict(color=[ut.univie_colors_100[c] for c in original_carrier_assignment], width=2),
                          color=ut.univie_colors_60[carrier]),
                      text=df['id_'], name=f'Carrier {carrier}, Tour {tour}',
                      hovertext=hover_text
                      # legendgroup=carrier, hoverinfo='x+y'
                      )


def _make_unrouted_scatter(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int):
    cs = solution.carrier_solutions[carrier]
    unrouted_vertices = ut.flatten([list(instance.pickup_delivery_pair(r)) for r in cs.unrouted_requests])

    df = pd.DataFrame({
        'id_': unrouted_vertices,
        'x': [instance.x_coords[v] for v in unrouted_vertices],
        'y': [instance.y_coords[v] for v in unrouted_vertices],
        'revenue': [instance.revenue[v] for v in unrouted_vertices],
        'tw_open': [solution.tw_open[v] for v in unrouted_vertices],
        'tw_close': [solution.tw_close[v] for v in unrouted_vertices],
        'original_carrier_assignment': [instance.request_to_carrier_assignment[r] for r in
                                        [instance.request_from_vertex(v) for v in unrouted_vertices]]})

    hover_text = []
    for v in unrouted_vertices:
        if v < instance.num_requests:
            v_type = 'Pickup'
        else:
            v_type = 'Delivery'
        hover_text.append(f'Request {instance.request_from_vertex(v)}</br>{v_type}</br>Revenue: {instance.revenue[v]}')

    return go.Scatter(
        x=df['x'], y=df['y'], mode='markers+text',
        marker=dict(
            symbol=['circle'] * len(unrouted_vertices),
            size=ut.linear_interpolation(df['revenue'].values, 15, 30),
            line=dict(color=[ut.univie_colors_100[c] for c in df['original_carrier_assignment']], width=2),
            color=ut.univie_colors_60[carrier]),
        text=df['id_'],
        textfont=dict(color='red'),
        name=f'Carrier {carrier}, unrouted',
        hovertext=hover_text
        # legendgroup=carrier, hoverinfo='x+y'
    )


def _make_tour_edges(instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, tour: int):
    t = solution.carrier_solutions[carrier].tours[tour]
    # creating arrows as annotations
    directed_edges = []
    for i in range(len(t.routing_sequence) - 1):
        from_vertex = t.routing_sequence[i]
        to_vertex = t.routing_sequence[i + 1]
        arrow_tail = (instance.x_coords[from_vertex], instance.y_coords[from_vertex])
        arrow_head = (instance.x_coords[to_vertex], instance.y_coords[to_vertex])
        min_rev = min(instance.revenue)
        max_rev = max(instance.revenue)

        anno = go.layout.Annotation(
            x=arrow_head[0], y=arrow_head[1], ax=arrow_tail[0], ay=arrow_tail[1],
            xref='x', yref='y', axref='x', ayref='y',
            text="",
            showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=1,
            arrowcolor=ut.univie_colors_100[carrier],
            standoff=ut.linear_interpolation([instance.revenue[to_vertex]], 15, 30, min_rev, max_rev)[0] / 2,
            startstandoff=ut.linear_interpolation([instance.revenue[from_vertex]], 15, 30, min_rev, max_rev)[0] / 2,
            name=f'Carrier {carrier}, Tour {t}')
        directed_edges.append(anno)
    return directed_edges


def _add_carrier_solution(fig: go.Figure, instance, solution: slt.GlobalSolution, carrier: int):
    carrier_solution = solution.carrier_solutions[carrier]
    scatter_traces = []
    edge_traces = []
    for tour in range(carrier_solution.num_tours()):
        tour_scatter = _make_tour_scatter(instance, solution, carrier, tour)
        fig.add_trace(tour_scatter)
        scatter_traces.append(tour_scatter)

        edges = _make_tour_edges(instance, solution, carrier, tour)
        for edge in edges:
            fig.add_annotation(edge)
        edge_traces.append(edges)

    if carrier_solution.unrouted_requests:
        unrouted_scatter = _make_unrouted_scatter(instance, solution, carrier)
        fig.add_trace(unrouted_scatter)
        scatter_traces.append(unrouted_scatter)

    return scatter_traces, edge_traces


def _make_unassigned_scatter(fig: go.Figure, instance: it.PDPInstance, solution: slt.GlobalSolution):
    vertex_id = ut.flatten(
        [[p, d] for p, d in [instance.pickup_delivery_pair(r) for r in solution.unassigned_requests]])
    df = pd.DataFrame({
        'id_': vertex_id,
        'x': [instance.x_coords[v] for v in vertex_id],
        'y': [instance.y_coords[v] for v in vertex_id],
        'revenue': [instance.revenue[v] for v in vertex_id],
        'original_carrier_assignment': [instance.request_to_carrier_assignment[r] for r in
                                        [instance.request_from_vertex(v) for v in vertex_id]]
    })
    hover_text = []
    for v in vertex_id:
        if v < instance.num_requests:
            v_type = 'Pickup'
        else:
            v_type = 'Delivery'
        hover_text.append(f'Request {instance.request_from_vertex(v)}</br>{v_type}</br>Revenue: {instance.revenue[v]}')

    return go.Scatter(x=df['x'], y=df['y'], mode='markers+text',
                      marker=dict(
                          symbol=['circle'] * len(df),
                          size=ut.linear_interpolation(df['revenue'].values, 15, 30),
                          line=dict(color=[ut.univie_colors_100[c] for c in df['original_carrier_assignment']],
                                    width=2),
                          color='white'),
                      text=df['id_'],
                      name=f'Unassigned',
                      hovertext=hover_text
                      )


def plot_solution_2(instance: it.PDPInstance, solution: slt.GlobalSolution, title='', tours: bool = True,
                    time_windows: bool = True, arrival_times: bool = True, service_times: bool = True,
                    show: bool = False):
    fig = go.Figure()
    # [[scatter tour 0, scatter tour 1], [scatter tour 0, scatter tour 1, scatter tour 2], ...]
    scatters = []
    # [carrier 0:
    #   tour 0: [[edge_0, edge_1, edge_2, ...],
    #   tour 1: [edge_0, edge_1, edge_2, ...], ],
    # carrier 1:
    #   tour 0: [[edge_0, edge_1, edge_2, ...],
    #   tour 1: [edge_0, edge_1, edge_2, ...], ]
    # ]
    edges = []

    for c in range(instance.num_carriers):
        scatter_traces, edge_traces = _add_carrier_solution(fig, instance, solution, c)
        scatters.append(scatter_traces)
        edges.append(edge_traces)

    if solution.unassigned_requests:
        unassigned_scatter = _make_unassigned_scatter(fig, instance, solution)
        fig.add_trace(unassigned_scatter)
        scatters.append(unassigned_scatter)

    # custom buttons to hide edges
    button_dicts = []
    for c in range(instance.num_carriers):
        for t in range(solution.carrier_solutions[c].num_tours()):
            button_dicts.append(
                dict(label=f'Carrier {c}, Tour {t}',
                     method='update',
                     args=[dict(visible=[True]),
                           dict(annotations=edges[c][t])
                           ],
                     # toggle on/off buttons -> not working as intended
                     # args2=[dict(visible=[True]),
                     #        dict(annotations=edges)
                     #        ],
                     )
            )
    button_dicts.append(
        dict(label=f'All',
             method='update',
             args=[dict(visible=[True]),
                   dict(annotations=ut.flatten(edges))
                   ],

             )
    )

    fig.update_layout(updatemenus=
                      [dict(type='buttons',
                            y=c / instance.num_carriers,
                            yanchor='auto',
                            active=-1,
                            buttons=button_dicts
                            )
                       ],
                      title=title,
                      # height=1200,
                      # width=1600
                      )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    if show:
        fig.show(config=config)


def plot_solution(instance: it.PDPInstance, solution: slt.GlobalSolution, title='', tours: bool = True,
                  time_windows: bool = True, arrival_times: bool = True, service_times: bool = True,
                  show: bool = False):
    vertices = list(range(instance.num_carriers + 2 * instance.num_requests))
    fig = _vertex_figure(instance, solution, vertices, title)
    for carrier in range(instance.num_carriers):
        for tour in range(len(solution.carrier_solutions[carrier].tours)):
            if tours:
                _add_edge_traces(fig, instance, solution.carrier_solutions[carrier].tours[tour].routing_sequence)
            if arrival_times or service_times:
                _add_tour_annotations(fig, instance, solution, carrier, tour)
    if time_windows:
        _add_vertex_annotations(fig, instance, solution, vertices)
    if show:
        fig.show(config=config)


class CarrierConstructionAnimation(object):
    def __init__(self, carrier, title=None):
        self.carrier = carrier
        self.fig: plt.Figure
        self.ax: plt.Axes
        self.fig, self.ax = plt.subplots()
        # self.fig.set_tight_layout(True)
        self.ax.grid()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title(title)
        x = [r.coords.x for r in carrier.requests]
        y = [r.coords.y for r in carrier.requests]
        self.requests_artist = self.ax.plot(x, y, marker='o', markersize=9, mfc='white', c='black', ls='')
        self.depot_artist = self.ax.plot(*carrier.depot.coords, marker='s', markersize=9, c='black', ls='',
                                         label=carrier.depot.id_)
        self.freeze_frames = [*self.requests_artist, *self.depot_artist]  # artists that are plotted each frame
        self.ims = []

    def add_current_frame(self, vehicle: Union['all',] = 'all'):
        if vehicle != 'all':
            artists = vehicle.tour.plot(color=vehicle.color, plot_depot=False)
        else:
            artists = []
            for v in self.carrier.active_vehicles:
                a = v.tour.plot(color=v.color, plot_depot=False)
                artists.extend(a)
        legend_artists = [a for a in artists if type(a) != Annotation]
        legend_artists.extend(self.depot_artist)
        self.ax: plt.Axes
        legend = self.ax.legend(handles=legend_artists,
                                labels=[la.get_label() for la in legend_artists],
                                # loc='upper left',
                                # bbox_to_anchor=(1, 1),
                                # fancybox=True,
                                # shadow=True,
                                # ncol=5,
                                # mode="expand",
                                # borderaxespad=0.
                                )
        frame = [*self.requests_artist, *self.freeze_frames, *artists, legend]
        self.ims.append(frame)
        return frame

    def freeze_current_frame(self, vehicle=None):
        freeze_frame = self.add_current_frame(vehicle)
        self.freeze_frames.append(freeze_frame)
        return freeze_frame

    def _finish(self, repeat_last_frame=80, interval=60, repeat=True, repeat_delay=300):
        self.ims.extend([self.ims[-1]] * repeat_last_frame)
        self.animation = ani.ArtistAnimation(self.fig,
                                             artists=self.ims,
                                             interval=interval,
                                             repeat=repeat,
                                             repeat_delay=repeat_delay,
                                             blit=True)

    def show(self):
        self._finish()
        plt.show()
        # self.fig.show()

    def save(self, filename=ut.path_output.joinpath('Animations', 'animation.gif')):
        self._finish()
        movie_writer = ani.ImageMagickFileWriter(fps=24)
        # filename.mkdir(parents=True, exist_ok=True)  # TODO use pathlib Paths everywhere!
        self.animation.save(filename, writer=movie_writer)
        return filename
