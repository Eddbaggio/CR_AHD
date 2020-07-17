import matplotlib.animation as ani
import matplotlib.pyplot as plt
from carrier import Carrier


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
        self.freeze_frames = []
        frame = [*self.requests, *self.depot]
        self.ims = [frame]

    def add_current_frame(self, vehicle=None):
        if vehicle is not None:
            artists = vehicle.tour.plot(c=vehicle.color)
        else:
            artists = []
            for v in self.carrier.vehicles:
                a = v.tour.plot(c=v.color)
                artists.extend(*a)
        frame = [*self.requests, *self.freeze_frames, *artists]
        self.ims.append(frame)
        return frame

    def freeze_current_frame(self, vehicle=None):
        freeze_frame = self.add_current_frame(vehicle)
        self.freeze_frames.append(freeze_frame)
        return freeze_frame

    def show(self, repeat_last_frame=100, interval=40, repeat=True, repeat_delay=500):
        self.ims.extend([self.ims[-1]] * repeat_last_frame)
        aa = ani.ArtistAnimation(self.fig,
                                 artists=self.ims,
                                 interval=interval,
                                 repeat=repeat,
                                 repeat_delay=repeat_delay)
        self.fig.show()
