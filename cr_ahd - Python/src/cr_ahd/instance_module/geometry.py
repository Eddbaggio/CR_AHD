import math
from typing import Union

import numpy as np
from shapely.geometry import Point, Polygon, LineString

# CONVENTION
# when using coordinates, use the order LATITUDE FIRST, LONGITUDE SECOND! All inputs will be rearranged to follow this
# this order. https://en.wikipedia.org/wiki/ISO_6709#Items


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


def flip_coords(obj: Union[Point, LineString, Polygon]):
    """
    flips all coordinates of the object. Useful if data is given in long, lat coordinates instead of lat, long.
    """
    if isinstance(obj, Point):
        return Point(obj.y, obj.x)
    if isinstance(obj, LineString):
        coords = []
        for c in obj.coords:
            coords.append((c[1], c[0]))
        return LineString(coords)
    if isinstance(obj, Polygon):
        if obj.interiors:
            raise NotImplementedError('flip_coords does not handle Polygons with holes yet')
        coords = []
        for c in obj.exterior.coords:
            coords.append((c[1], c[0]))
        return Polygon(coords)
    else:
        raise TypeError