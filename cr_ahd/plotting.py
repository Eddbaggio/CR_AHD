from typing import Union

import matplotlib.animation as ani
import matplotlib.pyplot as plt
from carrier import Carrier
from vehicle import Vehicle


class CarrierConstructionAnimation(object):
    def __init__(self, carrier: Carrier):
        self.carrier = carrier
        self.fig, self.ax = plt.subplots()
        self.fig: plt.Figure
        self.ax: plt.Axes
        self.ax.grid()
        x = [r.coords.x for _, r in carrier.requests.items()]
        y = [r.coords.y for _, r in carrier.requests.items()]
        self.requests = self.ax.plot(x, y, marker='o', markersize=9, mfc='white', c='black', ls='')
        self.depot = self.ax.plot(*carrier.depot.coords, marker='s', markersize=9, c='black')
        self.freeze_frames = [*self.requests, *self.depot]  # artists that are plotted each frame
        self.ims = []

    def add_current_frame(self, vehicle: Union['all', Vehicle] = 'all'):
        if vehicle != 'all':
            artists = vehicle.tour.plot(color=vehicle.color)
        else:
            artists = []
            for v in self.carrier.vehicles:
                if len(v.tour) > 2:
                    a = v.tour.plot(color=v.color, )
                    artists.extend(a)
                else:
                    continue  # skip empty vehicles
        frame = [*self.requests, *self.freeze_frames, *artists]
        self.ims.append(frame)
        return frame

    def freeze_current_frame(self, vehicle=None):
        freeze_frame = self.add_current_frame(vehicle)
        self.freeze_frames.append(freeze_frame)
        return freeze_frame

    def _finish(self, repeat_last_frame=80, interval=40, repeat=True, repeat_delay=300):
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

    def save(self, filename='../data/Output/Animations/animation.gif'):
        self._finish()
        movie_writer = ani.ImageMagickFileWriter(fps=24)
        self.animation.save(filename, writer=movie_writer)
        return filename
