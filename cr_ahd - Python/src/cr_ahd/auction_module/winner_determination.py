import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence

import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB

import utility_module.io
from core_module import instance as it, solution as slt
from utility_module import utils as ut

logger = logging.getLogger(__name__)


class WinnerDeterminationBehavior(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def execute(self,
                instance: it.MDPDPTWInstance,
                solution: slt.CAHDSolution,
                auction_pool,
                bundles: Sequence,
                bids_matrix: Sequence,
                original_bundling_labels: Sequence):
        """
        apply the concrete winner determination behavior. Each carrier can only win a single bundle for now and the
        number of bundles must be equal to the number of carriers

        :param bids_matrix: nested sequence of bids. the first axis is the bundles, the second axis (inner sequences)
         contain the carrier bids on that bundle
        """

        status, winner_bundles, bundle_winners = self._determine_winners(instance, solution, auction_pool, bundles,
                                                                         bids_matrix)

        # fall back to the pre-auction assignment
        if status != GRB.OPTIMAL:
            winner_bundles = ut.indices_to_nested_lists(original_bundling_labels, auction_pool)
            bundle_winners = [w for i, w in enumerate(original_bundling_labels) if
                              w not in original_bundling_labels[:i]]

        return status, winner_bundles, bundle_winners

    @abstractmethod
    def _determine_winners(self,
                           instance: it.MDPDPTWInstance,
                           solution: slt.CAHDSolution,
                           auction_pool: Sequence[int],
                           bundles: Sequence[Sequence[int]],
                           bids_matrix: Sequence[Sequence[float]]):
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


class MaxBidGurobiCAP1(WinnerDeterminationBehavior):
    """
    Set Partitioning Formulation for the Combinatorial Auction Problem / Winner Determination Problem
    Following Vries,S.de, & Vohra,R.V. (2003). Combinatorial Auctions: A Survey.
    https://doi.org/10.1287/ijoc.15.3.284.16077
    CAP1
    """

    def _determine_winners(self,
                           instance: it.MDPDPTWInstance,
                           solution: slt.CAHDSolution,
                           auction_pool: Sequence[int],
                           bundles: Sequence[Sequence[int]],
                           bids_matrix: Sequence[Sequence[float]]):
        # a numpy array
        bids_matrix = np.array(bids_matrix)
        bids_matrix[bids_matrix == -float('inf')] = -GRB.INFINITY

        # bids as a tuple dict
        coeff = {(b, c): bids_matrix[b, c] for b in range(len(bundles)) for c in range(instance.num_carriers)}

        # model
        m = gp.Model(name="Set Partitioning Problem")
        m.setParam('OutputFlag', 0)

        # variables: bundle-to-bidder assignment
        y: gp.tupledict = m.addVars(len(bundles), instance.num_carriers, vtype=GRB.BINARY, name='y')

        # objective: max the sum of realized bids
        m.setObjective(y.prod(coeff), GRB.MAXIMIZE)

        # constraints: each request is assigned to exactly one carrier
        # (set covering: assigned to *at least* one carrier
        # set packing: assigned to *at most* one carrier)
        for request in auction_pool:
            expr = gp.LinExpr()
            for b, bundle in enumerate(bundles):
                if request in bundle:
                    expr += y.sum(b, '*')
            m.addLConstr(expr, GRB.EQUAL, 1, f'single assignment {request}')

        # constraints: no carrier wins more than one bundle
        for c in range(instance.num_carriers):
            m.addLConstr(y.sum('*', c), GRB.LESS_EQUAL, 1, f'single bundle {c}')

        # write
        path = utility_module.io.output_dir.joinpath(f'{solution.num_carriers()}carriers',
                                                     solution.id_ + '_' + solution.solver_config['solution_algorithm'])
        path = utility_module.io.unique_path(path.parent, path.stem + 'WDP_#{:03d}.lp')
        path.parent.mkdir(parents=True, exist_ok=True)
        m.write(str(path))

        # solve
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None
        assert m.Status == GRB.OPTIMAL, f'Winner Determination is not optimal; status: {m.Status}'
        # status 3 is 'infeasible'

        winner_bundles = []
        bundle_winners = []
        logger.debug(f'the optimal solution for the Winner Determination Problem with {len(bundles)} bundles:')
        logger.debug(f'Objective: {m.objVal}')
        for b, bundle in enumerate(bundles):
            for c in range(instance.num_carriers):
                if y[b, c].x >= 0.99:
                    winner_bundles.append(bundle)
                    bundle_winners.append(c)
                    logger.debug(f'Bundle {b}: {bundle} assigned to {c} for a bid of {bids_matrix[b, c]}')

        return m.Status, winner_bundles, bundle_winners


class MaxBidGurobiCAP2(WinnerDeterminationBehavior):
    """
    Set Partitioning Formulation for the Combinatorial Auction Problem / Winner Determination Problem
    Following Vries,S.de, & Vohra,R.V. (2003). Combinatorial Auctions: A Survey.
    https://doi.org/10.1287/ijoc.15.3.284.16077
    CAP2
    """

    def _determine_winners(self,
                           instance: it.MDPDPTWInstance,
                           solution: slt.CAHDSolution,
                           auction_pool: Sequence[int],
                           bundles: Sequence[Sequence[int]],
                           bids_matrix: Sequence[Sequence[float]]):
        # (carrier, bid) tuples of the highest bid per bundle
        max_bidders = [ut.argmax(bundle_bids) for bundle_bids in bids_matrix]
        max_bids = [bids_matrix[b][max_bidders[b]] for b in range(len(bids_matrix))]

        # model
        m = gp.Model(name="Set Packing?/Covering?/Partitioning? Problem")
        m.setParam('OutputFlag', 0)

        # variables: is a bundle assigned to the max bidder or not?
        x = m.addVars(range(len(bundles)), vtype=GRB.BINARY, obj=max_bids, name='bundle assignment')

        # objective: max the sum of bids
        m.modelSense = GRB.MAXIMIZE

        # constraints: each request is assigned to exactly (!) one carrier --> partitioning
        # with at most one carrier --> set packing
        # with at least one --> set covering
        for request in auction_pool:
            m.addConstr(sum(x[i] for i in range(len(bundles)) if request in bundles[i]) == 1, "single assignment")

        # constraints: each carrier can win at most one bundle  # TODO check why this is not implemented!
        # for carrier in range(instance.num_carriers):

        # solve
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return m.Status, None, None
        assert m.Status == GRB.OPTIMAL, f'Winner Determination is not optimal; status: {m.Status}'
        # status 3 is 'infeasible'

        winner_bundles = []
        bundle_winners = []
        logger.debug(f'the optimal solution for the Winner Determination Problem with {len(bundles)} bundles:')
        logger.debug(f'Objective: {m.objVal}')
        for b_idx, bundle in enumerate(bundles):
            if x[b_idx].x > 0.99:
                winner_bundles.append(bundle)
                bundle_winners.append(max_bidders[b_idx])
                logger.debug(
                    f'Bundle {b_idx}: {bundle} assigned to {max_bidders[b_idx]} for a bid of {max_bids[b_idx]}')

        return winner_bundles, bundle_winners
