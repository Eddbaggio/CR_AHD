import argparse


# def custom_parser(string: str):
#     if '-' in string:
#         start, end = string.split('-')
#         return range(int(start, 10), int(end, 10) + 1)
#     else:
#         return [int(x, 10) for x in (string.split(','))]


parser = argparse.ArgumentParser(description='Optimizing CR_AHD instances')
""" these were for GH instances
# optional arguments
parser.add_argument('-x', '--run', help='runs of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-r', '--rad', help='radii of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-n', '--n', help='n of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-t', '--threads', help='number of threads to use',
                    type=int, default=1)
parser.add_argument('-f', '--fail', help='', action="store_true")
"""
# optional arguments
# t=vienna+d=7+c=3+n=10+o=100+r=08
parser.add_argument('-d', '--distance',
                    help='values for "distances of depots from city center" of vienna instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-c', '--num_carriers',
                    help='values for "number of carriers" of vienna instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-n', '--num_requests',
                    help='values for "num_customer_per_carrier" of vienna instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-v', '--carrier_max_num_tours',
                    help='values for "carrier_max_num_tours" of vienna instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-o', '--service_area_overlap',
                    help='values for "service_area_overlap" [0.0-1.0] of vienna instances that should be executed',
                    type=float, nargs='*', default=None)
parser.add_argument('-r', '--run',
                    help='runs of vienna instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-t', '--threads',
                    help='number of threads to use',
                    type=int, default=1)
parser.add_argument('-f', '--fail', help='shall the execution stop and fail if an error occurs?', action="store_true")
