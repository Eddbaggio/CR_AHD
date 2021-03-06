import datetime as dt
import itertools
from typing import List, Tuple, Dict, Sequence, Union

from auction_module import auction as au, bidding as bd, winner_determination as wd
from auction_module.request_selection import request_selection as rs, individual as rsi, bundle as rsb, neighbor as rsn
from auction_module.bundle_and_partition_valuation import partition_valuation as pv
from auction_module.bundle_generation import bundle_gen as bg, partition_based_bg as bgp
from routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from tw_management_module import tw_offering as two, tw_selection as tws, request_acceptance as ra


def configs():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """

    s_tour_construction: Sequence[cns.VRPTWInsertionConstruction] = [
        # cns.VRPTWMinTravelDistanceInsertion(),
        cns.VRPTWMinTravelDurationInsertion()
        # cns.MinTimeShiftInsertion()
    ]

    # t = float('inf') if ut.debugger_is_active() else 5
    t = 2
    s_tour_improvement: Sequence[mh.VRPTWMetaHeuristic] = [
        mh.NoMetaheuristic([nh.NoNeighborhood()], None),
        # mh.LocalSearchFirst([nh.VRPTWMoveDur()], 1),
        # mh.LocalSearchFirst([nh.VRPTWTwoOptDur()], 1),
        # mh.LocalSearchBest([nh.VRPTWMoveDur()], 1),
        # mh.VRPTWVariableNeighborhoodDescent([nh.VRPTWRelocateDur(), nh.VRPTWMoveDur(), nh.VRPTWTwoOptDurMax4()], t),
        # mh.VRPTWSequentialLocalSearch([nh.VRPTWTwoOptDur(), nh.VRPTWMoveDur(), nh.VRPTWRelocateDur()], t)
    ]

    s_max_num_accepted_infeasible: Sequence[int] = [0]

    s_request_acceptance_attractiveness: Sequence[ra.RequestAcceptanceAttractiveness] = [ra.Dummy()]

    s_time_window_length: Sequence[dt.timedelta] = [
        # dt.timedelta(hours=1),
        dt.timedelta(hours=2),
        # dt.timedelta(hours=4),
        # dt.timedelta(hours=8),
    ]

    s_time_window_offering: two.TWOfferingBehavior.__class__ = [
        two.FeasibleTW,
    ]

    s_time_window_selection: Sequence[tws.TWSelectionBehavior] = [
        tws.UnequalPreference(),
        # tws.UniformPreference(),
    ]

    # by setting this to an empty sequence, no collaborative solutions will be generated
    s_num_submitted_requests: Sequence[Union[int, float]] = [
        0.1,
        0.2,
    ]

    s_request_selection: Sequence[rs.RequestSelectionBehavior.__class__] = [
        rsi.Random,
        rsi.DepotDurations,
        rsi.MarginalCostProxy,
        rsi.EarlyTimeWindow,
        rsn.TemporalSpatialNeighbors,
    ]

    s_num_auction_bundles: Sequence[int] = [
        # 50,
        100,
        # 200,
        # 300,
        # 500
    ]

    s_bundle_generation: Sequence[Tuple[bg.BundleGenerationBehavior.__class__, Dict[str, float]]] = [
        (bgp.GeneticAlgorithm, dict(population_size=300,
                                    num_generations=100,
                                    mutation_rate=0.5,
                                    generation_gap=0.9, )
         ),
        # (bgp.BestOfAllPartitions, dict()),
        (bgp.RandomMaxKPartitions, dict())  # currently better than the GA
    ]

    s_partition_valuation: Sequence[pv.PartitionValuation.__class__] = [
        # bv.GHProxyPartitionValuation,
        # bv.MinDistancePartitionValuation,
        # bv.MinDurationPartitionValuation,
        pv.SumTravelDurationPartitionValuation,
        # bv.LosSchultePartitionValuation,
        # bv.RandomPartitionValuation,
    ]

    s_auction_policy: List[Dict] = [
        {'num_intermediate_auctions': 0, 'num_intermediate_auction_rounds': 0, 'num_final_auction_rounds': 1},
        # {'num_intermediate_auctions': 0, 'num_intermediate_auction_rounds': 0, 'num_final_auction_rounds': 2},
        # {'num_intermediate_auctions': 1, 'num_intermediate_auction_rounds': 1, 'num_final_auction_rounds': 1},
    ]

    # ===== Nested Parameter Loops =====
    for tour_construction, tour_improvement, max_num_accepted_infeasible, \
        request_acceptance_attractiveness, time_window_length, time_window_offering, \
        time_window_selection in itertools.product(s_tour_construction,
                                                   s_tour_improvement,
                                                   s_max_num_accepted_infeasible,
                                                   s_request_acceptance_attractiveness,
                                                   s_time_window_length,
                                                   s_time_window_offering,
                                                   s_time_window_selection
                                                   ):

        request_acceptance = ra.RequestAcceptanceBehavior(
            max_num_accepted_infeasible,
            request_acceptance_attractiveness,
            time_window_offering(time_window_length),
            time_window_selection
        )

        # ===== Isolated Planning Parameters, no auction =====
        isolated_planning = dict(
            request_acceptance=request_acceptance,
            tour_construction=tour_construction,
            tour_improvement=tour_improvement,
            num_intermediate_auctions=0,
            intermediate_auction=False,
            final_auction=False,
        )
        yield isolated_planning

        # auction-specific parameters
        for num_submitted_requests in s_num_submitted_requests:
            for request_selection in s_request_selection:
                for num_auction_bundles in s_num_auction_bundles:
                    for bundle_generation, bundle_generation_kwargs in s_bundle_generation:
                        for partition_valuation in s_partition_valuation:
                            for auction_policy in s_auction_policy:

                                # ===== INTERMEDIATE AUCTIONS =====
                                if auction_policy['num_intermediate_auctions'] > 0:
                                    assert num_submitted_requests % 2 == 0
                                    total_nsr_int = num_submitted_requests // 2
                                    total_nsr_fin = num_submitted_requests // 2

                                    assert total_nsr_int % auction_policy[
                                        'num_intermediate_auctions'] == 0
                                    nsr_int = total_nsr_int // auction_policy[
                                        'num_intermediate_auctions']

                                    assert nsr_int % auction_policy[
                                        'num_intermediate_auction_rounds'] == 0
                                    nsr_int_round = nsr_int // auction_policy[
                                        'num_intermediate_auction_rounds']

                                    intermediate_auction = au.Auction(
                                        tour_construction=tour_construction,
                                        tour_improvement=tour_improvement,
                                        request_selection=request_selection(nsr_int_round),
                                        bundle_generation=bundle_generation(
                                            # TODO is it fair to divide by num_int_auctions?
                                            num_auction_bundles=num_auction_bundles,
                                            # / auction_policy['num_intermediate_auctions'],
                                            partition_valuation=partition_valuation(),
                                            **bundle_generation_kwargs
                                        ),
                                        bidding=bd.ClearAndReinsertAll(tour_construction,
                                                                       tour_improvement
                                                                       ),
                                        winner_determination=wd.MaxBidGurobiCAP1(),
                                        # TODO add proper parameter
                                        num_auction_rounds=auction_policy[
                                            'num_intermediate_auction_rounds']
                                    )
                                else:
                                    total_nsr_fin = num_submitted_requests
                                    intermediate_auction = False

                                # ===== FINAL AUCTION =====
                                if auction_policy['num_final_auction_rounds'] > 1:
                                    assert total_nsr_fin % auction_policy[
                                        'num_final_auction_rounds'] == 0
                                    nsr_fin_round = total_nsr_fin // auction_policy[
                                        'num_final_auction_rounds']
                                else:
                                    nsr_fin_round = num_submitted_requests

                                final_auction = au.Auction(
                                    tour_construction=tour_construction,
                                    tour_improvement=tour_improvement,
                                    request_selection=request_selection(nsr_fin_round),
                                    bundle_generation=bundle_generation(
                                        # TODO is it fair to divide by num_int_auctions?
                                        num_auction_bundles=num_auction_bundles,
                                        # / auction_policy['num_intermediate_auctions'],
                                        partition_valuation=partition_valuation(),
                                        **bundle_generation_kwargs
                                    ),
                                    bidding=bd.ClearAndReinsertAll(tour_construction,
                                                                   tour_improvement
                                                                   ),
                                    winner_determination=wd.MaxBidGurobiCAP1(),
                                    num_auction_rounds=auction_policy[
                                        'num_final_auction_rounds']
                                )

                                collaborative_planning = dict(
                                    request_acceptance=request_acceptance,
                                    tour_construction=tour_construction,
                                    tour_improvement=tour_improvement,
                                    num_intermediate_auctions=auction_policy[
                                        'num_intermediate_auctions'],
                                    intermediate_auction=intermediate_auction,
                                    final_auction=final_auction,
                                )

                                yield collaborative_planning
