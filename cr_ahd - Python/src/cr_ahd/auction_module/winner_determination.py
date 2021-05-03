import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.cr_ahd.core_module import instance as it, solution as slt
from src.cr_ahd.utility_module import utils as ut

logger = logging.getLogger(__name__)


class WinnerDeterminationBehavior(ABC):
    def execute(self, instance: it.PDPInstance, solution: slt.GlobalSolution, bundles: Sequence, bids_matrix: Sequence):
        """
        apply the concrete winner determination behavior. Each carrier can only win a single bundle for now and the
        number of bundles must be equal to the number of carriers

        :param bids_matrix: nested sequence of bids. the first axis is the bundles, the second axis (inner sequences)
         contain the carrier bids on that bundle
        """

        return self._determine_winners(instance, solution, bundles, bids_matrix)

    @abstractmethod
    def _determine_winners(self, instance, solution, bundles, bids_matrix):
        """
        :param bids_matrix: nested sequence of bids. the first axis is the bundles, the second axis (inner sequences)
         contain the carrier bids on that bundle

        :return:
        """
        pass

    def _remove_bids_of_carrier(self, carrier, bids_matrix):
        for bundle_bids in bids_matrix:
            bundle_bids[carrier] = float('inf')
        pass


class MaxBidGurobi(WinnerDeterminationBehavior):
    """
    Set Packing Formulation for the Combinatorial Auction Problem / Winner Determination Problem
    """

    def _determine_winners(self, instance, solution, bundles, bids_matrix):
        # (carrier, bid) tuples of the highest bid per bundle
        max_bidders = [ut.argmax(bundle_bids) for bundle_bids in bids_matrix]
        max_bids = [bids_matrix[b][max_bidders[b]] for b in range(len(bids_matrix))]

        # model
        m = gp.Model(name="Set Packing/Covering/Partitioning Problem")
        m.setParam('OutputFlag', 0)

        # variables: is a bundle assigned to the max bidder or not?
        x = m.addVars(range(len(bundles)), vtype=GRB.BINARY, obj=max_bids, name='bundle assignment')

        # objective: max the sum of bids
        m.modelSense = GRB.MAXIMIZE

        # constraints: each request is assigned to at most one carrier
        # where set(ut.flatten(bundles)) is basically the pool of submitted requests (set() does not preserve order)
        for request in set(ut.flatten(bundles)):
            m.addConstr(sum(x[i] for i in range(len(bundles)) if request in bundles[i]) == 1, "single assignment")

        # solve
        m.optimize()

        winner_bundles = []
        bundle_winners = []
        logger.debug(f'the optimal solution for the Winner Determination Problem with {len(bundles)} bundles:')
        logger.debug(f'Objective: {m.objVal}')
        for b_idx, bundle in enumerate(bundles):
            if x[b_idx].x > 0.99:
                winner_bundles.append(bundle)
                bundle_winners.append(max_bidders[b_idx])
                logger.debug(f'Bundle {b_idx}: {bundle} assigned to {max_bidders[b_idx]} for a bid of {max_bids[b_idx]}')

        return winner_bundles, bundle_winners
