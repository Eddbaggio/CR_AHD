from typing import Union, List
import pandas as pd
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import plotly.express as px
import plotly.graph_objects as go
from src.cr_ahd.core_module.carrier import Carrier
from src.cr_ahd.core_module.vehicle import Vehicle
import src.cr_ahd.core_module.vertex as vx
from src.cr_ahd.utility_module.utils import path_output

config = dict({'scrollZoom': True})

def _vertex_figure(vertices: List[vx.BaseVertex], title: str = 'Scatter Plot - Vertices'):
    df = pd.DataFrame([v.coords for v in vertices], columns=['x', 'y'])
    df['id_'] = [v.id_ for v in vertices]
    df['tw_open'] = [v.tw.e for v in vertices]
    df['tw_close'] = [v.tw.l for v in vertices]
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


def _add_vertex_annotations(fig: go.Figure, vertices: List[vx.BaseVertex]):
    """creates annotations for the time window of each vertex"""
    for v in vertices:
        time_format = "Day %d %H:%M:%S"
        fig.add_annotation(x=v.coords.x, y=v.coords.y,
                           text=f'[{v.tw.e.strftime(time_format)} - {v.tw.l.strftime(time_format)}]',
                           clicktoshow='onoff',
                           yshift=-25, showarrow=False)
    pass


def _add_tour_annotations(fig: go.Figure, tour, arrival_time=True, service_time=True):
    for index, vertex in enumerate(tour.routing_sequence):
        time_format = "Day %d %H:%M:%S"
        text = ''
        if arrival_time:
            text += f'Arrival: {tour.arrival_schedule[index].strftime(time_format)}'
        if service_time:
            text += f'<br>Service: {tour.service_schedule[index].strftime(time_format)}'
            fig.add_annotation(x=vertex.coords.x, y=vertex.coords.y,
                               text=text,
                               clicktoshow='onoff',
                               yshift=-50, showarrow=False)
    pass


def _add_edge_traces(fig: go.Figure, vertices: List[vx.BaseVertex]):
    for i in range(len(vertices) - 1):
        arrow_tail = vertices[i].coords
        arrow_head = vertices[i + 1].coords
        fig.add_annotation(x=arrow_head.x,
                           y=arrow_head.y,
                           ax=arrow_tail.x,
                           ay=arrow_tail.y,
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


def plot_vertices(vertices: List[vx.BaseVertex], title: str, annotations=False, edges: bool = False,
                  show: bool = False):
    fig = _vertex_figure(vertices, title)
    if annotations:
        _add_vertex_annotations(fig, vertices)
    if edges:
        _add_edge_traces(fig, vertices)
    if show:
        fig.show(config=config)
    return fig


def plot_tour(tour, title, time_windows: bool = True, arrival_times: bool = True, service_times: bool = True,
              show: bool = False):
    fig = _vertex_figure(tour.routing_sequence, title)
    _add_edge_traces(fig, tour.routing_sequence)
    if time_windows:
        _add_vertex_annotations(fig, tour.routing_sequence)
    if arrival_times or service_times:
        _add_tour_annotations(fig, tour, arrival_times, service_times)
    if show:
        fig.show(config=config)
    return fig


def plot_carrier(carrier, title='', tours: bool = True, time_windows: bool = True, arrival_times: bool = True,
                 service_times: bool = True, show: bool = False):
    fig = _vertex_figure([carrier.depot, *carrier.vertices()], title)
    if tours:
        for vehicle in carrier.active_vehicles:
            _add_edge_traces(fig, vehicle.tour.routing_sequence)
    if time_windows:
        _add_vertex_annotations(fig, carrier.vertices())
    if arrival_times or service_times:
        for vehicle in carrier.active_vehicles:
            _add_tour_annotations(fig, vehicle.tour, arrival_times, service_times)
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

    def save(self, filename=path_output.joinpath('Animations', 'animation.gif')):
        self._finish()
        movie_writer = ani.ImageMagickFileWriter(fps=24)
        # filename.mkdir(parents=True, exist_ok=True)  # TODO use pathlib Paths everywhere!
        self.animation.save(filename, writer=movie_writer)
        return filename
