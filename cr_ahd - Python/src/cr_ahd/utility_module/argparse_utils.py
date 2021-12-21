import argparse
import re
from utility_module.io import instance_selector


def custom_parser(string: str):
    if string == '*':
        return string
    if '-' in string:
        start, end = string.split('-')
        return range(int(start, 10), int(end, 10) + 1)
    else:
        return [int(x, 10) for x in (string.split(','))]


parser = argparse.ArgumentParser(description='Optimizing CR_AHD instances')
parser.add_argument('--run', type=custom_parser, required=False, dest='run', )
parser.add_argument('--rad', type=custom_parser, required=False, dest='rad', )
parser.add_argument('--n', type=custom_parser, required=False, dest='n', )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=custom_parser)
    parser.add_argument('rad', type=custom_parser)
    parser.add_argument('n', type=custom_parser)

    # args = parser.parse_args(['4', '150', '*'])
    args = parser.parse_args()

    paths = instance_selector(args.run, args.rad, args.n)
    for p in paths:
        print(p.name)
