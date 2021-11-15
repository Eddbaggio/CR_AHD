import webbrowser
from typing import List

import folium
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate

from address_retrieval import read_vienna_addresses
import datetime as dt
import math
import geopandas as gp
import pandas as pd
from folium_plot import plot_points_in_vienna
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pyproj
import itertools

from utility_module import io


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
    new_lon = center_long + (x_offset_km / radius_earth_km) * (180 / math.pi) / math.cos(center_lat * math.pi / 180)
    return Point(new_lat, new_lon)


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


def generate_vienna_CR_AHD_instance(
        num_carriers: int = 3,
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
        carrier_competition_level: int = 2,
        # TODO make this a value that represent properly the service area overlap --> see formula i derived with Anna
):
    if num_carriers < 2:
        return

    circle_radius_km = 7  # radius of the circle on which the depots lie, in km
    vienna_lat, vienna_long = 48.210033, 16.363449

    depots = circle(vienna_lat, vienna_long, 7, resolution=num_carriers)
    depots = [Point(lat, lon) for lat, lon in list(depots.exterior.coords)[:-1]]
    service_areas = [circle(p.x, p.y, 7, 24) for p in depots]
    m = folium.Map(
        (vienna_lat, vienna_long),
        zoom_start=12,
        crs='EPSG3857',
        tiles='Stamen Toner'
    )

    # service_areas = gp.GeoDataFrame({'geometry': depots.to_crs(epsg=3857).buffer(7000, resolution=16)})
    # service_areas.to_crs(epsg=4326, inplace=True)

    # # compute geodesic distances to make sure the depot generation is correct:
    # for name, depot in depots.iterrows():
    #     line_string = LineString([vienna_center, depot.geometry])
    #     geod_epsg_4326 = pyproj.CRS("epsg:4326").get_geod()
    #     total_length = geod_epsg_4326.geometry_length(line_string)
    #     print(f'geodesic distance from center to {name}: {total_length:.3f}')

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

    # folium.Marker((vienna_lat, vienna_long), 'Center').add_to(m)
    # rad = folium.Circle((vienna_lat, vienna_long), 7000, color='red').add_to(m)

    # write and display
    path = io.output_dir.joinpath('folium_map.html')
    m.save(path.as_posix())
    webbrowser.open(path)

    # vienna_addresses = read_vienna_addresses()
    # vienna_addresses_with_depot = vienna_addresses.sjoin(service_areas, how='left', predicate='contains')

    return


if __name__ == '__main__':
    generate_vienna_CR_AHD_instance(num_carriers=3)
