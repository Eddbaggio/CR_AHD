from typing import Union

import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from carrier import Carrier
from vehicle import Vehicle
from pathlib import Path
from utils import path_output


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
        x = [r.coords.x for _, r in carrier.requests.items()]
        y = [r.coords.y for _, r in carrier.requests.items()]
        self.requests = self.ax.plot(x, y, marker='o', markersize=9, mfc='white', c='black', ls='')
        self.depot = self.ax.plot(*carrier.depot.coords, marker='s', markersize=9, c='black', ls='',
                                  label=carrier.depot.id_)
        self.freeze_frames = [*self.requests, *self.depot]  # artists that are plotted each frame
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
        legend_artists.extend(self.depot)
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
        frame = [*self.requests, *self.freeze_frames, *artists, legend]
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
