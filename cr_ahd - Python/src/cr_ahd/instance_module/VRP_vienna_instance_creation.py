import webbrowser
from typing import List

import folium
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate

from vienna_data_handling import read_vienna_addresses
import datetime as dt
import math
import geopandas as gp
import pandas as pd
from folium_plot import plot_points_in_vienna
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go
from mpl_toolkits import mplot3d

import pyproj
import itertools

from utility_module import io


def two_circle_overlap_area(radius=1, center_distance=1, ):
    # if the circles do NOT overlap, return the negative difference of circle diameter and center distance
    if center_distance * 0.5 >= radius:
        return 2 * radius - center_distance

    area_circle = np.pi * radius ** 2

    if center_distance == 0:
        return area_circle

    intersect_x = center_distance * 0.5
    intersect_y = np.sqrt(radius ** 2 - intersect_x ** 2)

    area_triangle = intersect_x * intersect_y
    alpha = np.degrees(np.arctan(intersect_y / intersect_x))
    area_pizza = 2 * (alpha / 360) * area_circle
    area_intersection = 2 * (area_pizza - area_triangle)
    # print(f'The area of the intersection of two circles with radius '
    #       f'{radius} that are {center_distance} units apart is {area_intersection}')
    return area_intersection


def point_on_circle(center_lat: float, center_long: float, radius_km: float, angle_degree: float):
    """
    create a point's lat long coordinates that is radius_km kilometers away from a center point in a direction given
    by the angle_degree.

    :returns: a shapely.Point object with lat, lon coordinates

    formulas taken from:
    https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    """
    radius_earth_km = 6378
    x_offset_km = math.cos(math.radians(angle_degree)) * radius_km
    y_offset_km = math.sin(math.radians(angle_degree)) * radius_km
    new_lat = center_lat + (y_offset_km / radius_earth_km) * (180 / math.pi)
    new_long = center_long + (x_offset_km / radius_earth_km) * (180 / math.pi) / math.cos(center_lat * math.pi / 180)
    return Point(new_lat, new_long)


def circle(center_lat: float, center_long: float, radius_km: float, resolution: int = 16):
    """
    create a circular polygon with a given radius. The resolution defines the number of points on the circumference
    of the circle
    """
    assert resolution > 2, "circle's circumference must have more than two points!"

    points = []
    for i in range(resolution):
        p = point_on_circle(center_lat, center_long, radius_km, (360 / resolution) * i)
        points.append(p)

    return Polygon(points)


def competition_level_to_service_radius(dist_between_carriers: float, comp_level: float = 0.5):
    assert comp_level < 1.0, f'for 100% competition, the radius would be infinite'
    assert comp_level > 0.0, f'for 0% competition, the radius would be zero'
    rad =


def generate_vienna_cr_ahd_instance(
        num_carriers: int = 3,
        dist_center_to_carrier_km=7,
        carrier_competition_level: int = 2,
        num_requests: int = 10,
        max_num_tours_per_carrier: int = 3,
        max_vehicle_load: int = 10,
        max_tour_length: int = 1000,  # TODO set appropriately
        # max_tour_duration: dt.timedelta = dt.timedelta(hours=8),
        requests_service_duration: List[dt.timedelta] = (dt.timedelta(minutes=4),) * (3 * 10),
        requests_revenue: List[int] = (99,) * (3 * 10),  # TODO should this depend on some distance?
        requests_load: List[int] = (1,) * (3 * 10),  # TODO should this be randomly distribute?
        # time_horizon_start: dt.datetime = dt.datetime(0, 0, 0, 9),
        # time_horizon_end: dt.datetime = dt.datetime(0, 0, 0, 17),
        # TODO make this a value that represent properly the service area overlap --> see formula i derived with Anna
):
    if num_carriers < 2:
        return

    vienna_lat, vienna_long = 48.210033, 16.363449

    depots = circle(vienna_lat, vienna_long, dist_center_to_carrier_km, resolution=num_carriers)
    depots = [Point(lat, long) for lat, long in list(depots.exterior.coords)[:-1]]
    service_areas = [circle(p.x, p.y, 7, 24) for p in depots]
    m = folium.Map(
        (vienna_lat, vienna_long),
        zoom_start=12,
        crs='EPSG3857',
        tiles='Stamen Toner'
    )

    cmap = plt.get_cmap('jet', num_carriers)

    # plot service areas
    for idx, (depot, service_area) in enumerate(zip(depots, service_areas)):
        folium.Polygon(locations=list(service_area.exterior.coords),
                       popup=f'Depot {idx}',
                       color=to_hex(cmap(idx / num_carriers)),
                       fill_color=to_hex(cmap(idx / num_carriers)),
                       fill_opacity=0.3,
                       ).add_to(m)
    # plot depots on top
    for idx, (depot, service_area) in enumerate(zip(depots, service_areas)):
        folium.CircleMarker(location=(depot.x, depot.y),
                            popup=f'Depot {idx}',
                            radius=5,
                            color=to_hex(cmap(idx / num_carriers)),
                            fill_color=to_hex(cmap(idx / num_carriers)),
                            fill_opacity=1
                            ).add_to(m)

    # city center
    # folium.Marker((vienna_lat, vienna_long), 'Center').add_to(m)
    # rad = folium.Circle((vienna_lat, vienna_long), 7000, color='red').add_to(m)

    # write and display
    # path = io.output_dir.joinpath('folium_map.html')
    # m.save(path.as_posix())
    # webbrowser.open(path)

    vienna_addresses = read_vienna_addresses(nrows=100)
    for carrier in range(num_carriers):
        vienna_addresses[f'carrier_{carrier}'] = vienna_addresses['geometry'].apply(
            lambda x: x.within(service_areas[carrier]))

    return


def plot_coords(objs):
    ax: plt.Axes
    fig, ax = plt.subplots()

    for obj in objs:
        if isinstance(obj, Point):
            coords = list(obj.coords)
            color = 'blue'
        elif isinstance(obj, Polygon):
            coords = list(obj.exterior.coords)
            color = 'red'
            # ax.plot(zip(coords))
        for x, y in coords:
            ax.scatter(x, y, c=color)
    plt.show()


if __name__ == '__main__':
    # generate_vienna_cr_ahd_instance(num_carriers=3)

    # generate a 3d plot to visualize the ratios between radius, center distance and intersection_area of two circles
    """
    r = np.linspace(0, 10, 50, endpoint=False)
    d = np.linspace(0, 10, 50, endpoint=False)
    R, D = np.meshgrid(r, d)
    vfunc = np.vectorize(two_circle_overlap_area)
    I = vfunc(R, D)

    fig = go.Figure(data=[go.Surface(x=R, y=D, z=I, colorscale='Viridis', connectgaps=False)], )
    fig.update_layout(scene=dict(
        xaxis_title='x=radius',
        yaxis_title='y=center distance',
        zaxis_title='z=intersection area',
    ),

        # width=700,
        # margin=dict(r=20, b=10, l=10, t=10)
        )
    fig.show()
    """
    r = np.linspace(0, 10, 100, endpoint=False)
    d = 7
    vfunc = np.vectorize(two_circle_overlap_area)
    i = vfunc(r, d) / (np.pi * r ** 2)
    fig = go.Figure(data=[go.Scatter(x=r, y=i, mode='lines+markers')])
    fig.update_layout(xaxis_title='radius', yaxis_title='intersection_area / circle_area')
    fig.show()
