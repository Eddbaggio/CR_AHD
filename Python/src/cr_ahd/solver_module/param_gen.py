from typing import List, Tuple, Dict

from auction_module import auction as au, \
    request_selection as rs, \
    bundle_generation as bg, \
    bidding as bd, \
    winner_determination as wd, \
    bundle_valuation as bv
from routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from tw_management_module import tw_offering as two, tw_selection as tws, request_acceptance as ra


def parameter_generator():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """

    tour_constructions: List[cns.VRPTWInsertionConstruction] = [
        # cns.VRPTWMinTravelDistanceInsertion(),
        cns.VRPTWMinTravelDurationInsertion()
        # cns.MinTimeShiftInsertion()
    ]

    acceptance_policies: List[Dict] = [
        {'max_num_accepted_infeasible': m, 'request_acceptance_attractiveness': r}
        for m, r in [
            (0, ra.Dummy()),
            # (1, ra.CloseToCompetitors50()),
            # (2, ra.CloseToCompetitors50()),
            # (2, ra.CloseToCompetitors75()),
            # (3, ra.CloseToCompetitors50()),
            # (4, ra.CloseToCompetitors50())
        ]]

    time_window_offerings = [
        two.FeasibleTW(),
        # two.NoTw()
    ]

    time_window_selections = [
        tws.UnequalPreference(),
        # tws.NoTW(),
    ]

    tour_improvements: List[mh.VRPTWMetaHeuristic] = [
        mh.NoMetaheuristic([nh.NoNeighborhood()], None),
        mh.LocalSearchFirst([nh.VRPTWMoveDur()], 1),
        mh.LocalSearchFirst([nh.VRPTWTwoOptDur()], 1),
        # mh.LocalSearchBest([nh.VRPTWMoveDur()], 1),
    ]

    nums_submitted_requests: List[int] = [
        # 3,
        4,
        # 5
    ]

    request_selections: List[rs.RequestSelectionBehavior.__class__] = [
        # rs.Random,
        # rs.SpatialBundleDSum,  # the original 'cluster' strategy by Gansterer & Hartl (2016)
        # rs.SpatialBundleDMax,
        # rs.MinDistanceToForeignDepotDMin,
        # rs.MarginalProfitProxy,
        # rs.MarginalProfitProxyNeighbor,
        # rs.ComboRaw,
        rs.ComboDistStandardized,
        # rs.LosSchulteBundle,
        # rs.TemporalRangeCluster,
        # TODO SpatioTemporalCluster is not yet good enough & sometimes even infeasible
        # rs.SpatioTemporalCluster,
        # rs.InfeasibleFirstRandomSecond,
    ]

    nums_auction_bundles: List[int] = [
        # 50,
        100,
        # 200,
        # 300,
        # 500
    ]

    bundle_generations: List[Tuple[bg.LimitedBundlePoolGenerationBehavior.__class__, Dict[str, float]]] = [
        (bg.GeneticAlgorithm, dict(population_size=300,
                                   num_generations=100,
                                   mutation_rate=0.5,
                                   generation_gap=0.9, )
         ),
        # (bg.BestOfAllBundlings, dict()),
        # (bg.RandomMaxKPartition, dict())
    ]

    bundling_valuations: List[bv.BundlingValuation.__class__] = [
        # bv.GHProxyBundlingValuation,
        # bv.MinDistanceBundlingValuation,
        bv.LosSchulteBundlingValuation,
        # bv.RandomBundlingValuation,
    ]

    auction_policies: List[Dict] = [
        {'num_intermediate_auctions': 0, 'num_intermediate_auction_rounds': 0, 'num_final_auction_rounds': 1},
        # {'num_intermediate_auctions': 0, 'num_intermediate_auction_rounds': 0, 'num_final_auction_rounds': 2},
        # {'num_intermediate_auctions': 1, 'num_intermediate_auction_rounds': 1, 'num_final_auction_rounds': 1},
    ]

    # ===== Nested Parameter Loops =====
    for tour_construction in tour_constructions:
        for tour_improvement in tour_improvements:
            for time_window_offering, time_window_selection in zip(time_window_offerings, time_window_selections):
                for acceptance_policy in acceptance_policies:
                    # ===== Isolated Planning Parameters, no auction =====
                    isolated_planning = dict(
                        request_acceptance=ra.RequestAcceptanceBehavior(
                            acceptance_policy['max_num_accepted_infeasible'],
                            acceptance_policy['request_acceptance_attractiveness'],
                            time_window_offering, time_window_selection),
                        tour_construction=tour_construction,
                        tour_improvement=tour_improvement,
                        num_intermediate_auctions=0,
                        intermediate_auction=False,
                        final_auction=False,
                    )
                    yield isolated_planning

                    # auction-specific parameters
                    for num_submitted_requests in nums_submitted_requests:
                        for request_selection in request_selections:
                            for num_auction_bundles in nums_auction_bundles:
                                for bundle_generation, bundle_generation_kwargs in bundle_generations:
                                    for bundling_valuation in bundling_valuations:
                                        for auction_policy in auction_policies:

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
                                                        bundling_valuation=bundling_valuation(),
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
                                            assert total_nsr_fin % auction_policy[
                                                'num_final_auction_rounds'] == 0
                                            nsr_fin_round = total_nsr_fin // auction_policy[
                                                'num_final_auction_rounds']

                                            final_auction = au.Auction(
                                                tour_construction=tour_construction,
                                                tour_improvement=tour_improvement,
                                                request_selection=request_selection(nsr_fin_round),
                                                bundle_generation=bundle_generation(
                                                    # TODO is it fair to divide by num_int_auctions?
                                                    num_auction_bundles=num_auction_bundles,
                                                    # / auction_policy['num_intermediate_auctions'],
                                                    bundling_valuation=bundling_valuation(),
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
                                                request_acceptance=ra.RequestAcceptanceBehavior(
                                                    acceptance_policy['max_num_accepted_infeasible'],
                                                    acceptance_policy['request_acceptance_attractiveness'],
                                                    time_window_offering, time_window_selection),
                                                tour_construction=tour_construction,
                                                tour_improvement=tour_improvement,
                                                num_intermediate_auctions=auction_policy[
                                                    'num_intermediate_auctions'],
                                                intermediate_auction=intermediate_auction,
                                                final_auction=final_auction,
                                            )

                                            yield collaborative_planning
