import argparse


# def custom_parser(string: str):
#     if '-' in string:
#         start, end = string.split('-')
#         return range(int(start, 10), int(end, 10) + 1)
#     else:
#         return [int(x, 10) for x in (string.split(','))]


parser = argparse.ArgumentParser(description='Optimizing CR_AHD instances')
# optional argument
parser.add_argument('-x', '--run', help='runs of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-r', '--rad', help='radii of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-n', '--n', help='n of GH instances that should be executed',
                    type=int, nargs='*', default=None)
parser.add_argument('-t', '--threads', help='number of threads to use',
                    type=int, default=1)
parser.add_argument('-f', '--fail', help='', action="store_true")
