from abc import abstractmethod, ABC
import logging
import random
from copy import deepcopy
from math import exp, log
from typing import Sequence

from src.cr_ahd.auction_module import bundle_valuation as bv
from src.cr_ahd.core_module import instance as it, solution as slt, tour as tr


def shaw_removal(instance: it.PDPInstance, solution: slt.CAHDSolution, num_removal_requests: int = 1, p: int = 1000):
    """
    removes num_removal_requests requests from the solution. selection is based on a similarity measure and some
    randomness that is introduced by the parameter p. removal happens in place

    :param instance:
    :param solution:
    :param num_removal_requests:
    :param p: introduces randomness to the selection of requests. a low value of p corresponds to much randomness
    :return: the removed requests as a list
    """
    assert p >= 1
    assert num_removal_requests <= instance.num_requests

    # select random initial request
    fixed_requests = []
    for carrier_ in solution.carriers:
        fixed_requests.extend(carrier_.accepted_requests)
    r = random.choice(fixed_requests)

    removal_requests = [r]
    fixed_requests.remove(r)

    # select similar requests
    while len(removal_requests) < num_removal_requests:
        r = random.choice(removal_requests)
        fixed_requests.sort(
            key=lambda x: bv.ropke_pisinger_request_similarity(instance, solution, r, x, capacity_weight=0))
        removal_request_index = round(random.random() ** p * len(fixed_requests))
        removal_requests.append(fixed_requests[removal_request_index])#

    # remove


    return removal_requests
