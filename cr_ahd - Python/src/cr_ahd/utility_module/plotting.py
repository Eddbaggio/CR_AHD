from typing import Union, List
import pandas as pd
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import plotly.express as px
import plotly.graph_objects as go
from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.vehicle import Vehicle
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


def _add_tour(fig: go.Figure, instance: it.PDPInstance, solution: slt.GlobalSolution, carrier: int, t: int):
    tour = solution.carrier_solutions[carrier].tours[t]

    df = pd.DataFrame({
        'id_': tour.routing_sequence[:-1],
        'x': [instance.x_coords[v] for v in tour.routing_sequence[:-1]],
        'y': [instance.y_coords[v] for v in tour.routing_sequence[:-1]],
        'tw_open': [solution.tw_open[v] for v in tour.routing_sequence[:-1]],
        'tw_close': [solution.tw_close[v] for v in tour.routing_sequence[:-1]], })
    fig.add_scatter(x=df['x'], y=df['y'], mode='markers+text',
                    marker_symbol=['square', *['circle'] * (tour.num_routing_stops - 2), 'square'],
                    marker_size=20,
                    marker_color=ut.univie_colors_60[carrier], marker_line_color=ut.univie_colors_100[carrier],
                    marker_line_width=2,
                    text=df['id_'], name=f'Carrier {carrier}, Tour {t}',
                    # legendgroup=carrier, hoverinfo='x+y'
                    )
    for i in range(len(tour.routing_sequence) - 1):
        arrow_tail = (instance.x_coords[tour.routing_sequence[i]], instance.y_coords[tour.routing_sequence[i]])
        arrow_head = (instance.x_coords[tour.routing_sequence[i + 1]], instance.y_coords[tour.routing_sequence[i + 1]])
        fig.add_annotation(x=arrow_head[0], y=arrow_head[1], ax=arrow_tail[0], ay=arrow_tail[1],
                           xref='x', yref='y', axref='x', ayref='y',
                           text="",
                           showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=1,
                           arrowcolor=ut.univie_colors_100[carrier],
                           standoff=10, startstandoff=10, name=f'Carrier {carrier}, Tour {tour}')
    pass


def _add_carrier_solution(fig, instance, solution: slt.GlobalSolution, carrier: int):
    carrier_solution = solution.carrier_solutions[carrier]
    for t in range(carrier_solution.num_tours()):
        _add_tour(fig, instance, solution, carrier, t)


def plot_solution_2(instance: it.PDPInstance, solution: slt.GlobalSolution, title='', tours: bool = True,
                    time_windows: bool = True, arrival_times: bool = True, service_times: bool = True,
                    show: bool = False):
    fig = go.Figure()
    for c in range(instance.num_carriers):
        _add_carrier_solution(fig, instance, solution, c)
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
    def __init__(self, carrier: Carrier, title=None):
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

    def add_current_frame(self, vehicle: Union['all', Vehicle] = 'all'):
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
