from typing import List, Tuple, Dict

from routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from tw_management_module import tw_offering as two, tw_selection as tws
from auction_module import auction as au,\
    request_selection as rs,\
    bundle_generation as bg, \
    bidding as bd,\
    winner_determination as wd,\
    bundle_valuation as bv


def parameter_generator():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """

    tour_constructions: List[cns.PDPParallelInsertionConstruction] = [
        cns.MinTravelDistanceInsertion(),
        # cns.MinTimeShiftInsertion()
    ]

    time_window_offerings = [
        two.FeasibleTW(),
        # two.NoTw()
    ]

    time_window_selections = [
        tws.UnequalPreference(),
        # tws.NoTW(),
    ]

    tour_improvements: List[mh.PDPTWMetaHeuristic.__class__] = [
        # mh.PDPTWSequentialLocalSearch,
        # mh.PDPTWIteratedLocalSearch,
        mh.PDPTWVariableNeighborhoodDescent,
        # mh.PDPTWReducedVariableNeighborhoodSearch,
        # mh.PDPTWVariableNeighborhoodSearch,
        # mh.PDPTWSimulatedAnnealing,
        # mh.NoMetaheuristic,
    ]

    neighborhood_collections: List[List[nh.Neighborhood]] = [
        [nh.PDPMove(), nh.PDPTwoOpt(), nh.PDPRelocate()],
    ]

    tour_improvement_time_limits: List[float] = [
        1,
        # 2,
        # 5,
        # 10
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
        rs.ComboStandardized,
        # rs.LosSchulteBundle,
        # rs.TemporalRangeCluster,
        # TODO SpatioTemporalCluster is not yet good enough & sometimes even infeasible
        # rs.SpatioTemporalCluster,
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
        bv.GHProxyBundlingValuation,
        # bv.MinDistanceBundlingValuation,
        # bv.LosSchulteBundlingValuation,
        # bv.RandomBundlingValuation,
    ]

    nums_final_auction_rounds: List[int] = [
        1,
        2,
    ]

    # ===== Nested Parameter Loops =====
    for tour_construction in tour_constructions:
        for tour_improvement in tour_improvements:
            for neighborhoods in neighborhood_collections:
                for tour_improvement_time_limit in tour_improvement_time_limits:
                    for time_window_offering, time_window_selection in zip(time_window_offerings,
                                                                           time_window_selections):
                        # ===== Isolated Planning Parameters, no auction =====
                        isolated_planning = dict(
                            time_window_offering=time_window_offering,
                            time_window_selection=time_window_selection,
                            tour_construction=tour_construction,
                            tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
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
                                            for num_final_auction_rounds in nums_final_auction_rounds:
                                                # ===== final_auction for collaborative planning =====
                                                final_auction = au.Auction(
                                                    tour_construction=tour_construction,
                                                    tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
                                                    request_selection=request_selection(num_submitted_requests),
                                                    bundle_generation=bundle_generation(
                                                        num_auction_bundles=num_auction_bundles,
                                                        bundling_valuation=bundling_valuation(),
                                                        **bundle_generation_kwargs
                                                    ),
                                                    bidding=bd.ClearAndReinsertAll(
                                                        tour_construction,
                                                        tour_improvement(neighborhoods, tour_improvement_time_limit)
                                                    ),
                                                    winner_determination=wd.MaxBidGurobiCAP1(),
                                                    num_auction_rounds=num_final_auction_rounds
                                                )

                                                # collaborative planning with only a final auction
                                                collaborative_planning_final = dict(
                                                    time_window_offering=time_window_offering,
                                                    time_window_selection=time_window_selection,
                                                    tour_construction=tour_construction,
                                                    tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
                                                    num_intermediate_auctions=0,
                                                    intermediate_auction=False,
                                                    final_auction=final_auction,
                                                )
                                                yield collaborative_planning_final

                                            for num_intermediate_auctions in range(1, 2):  # TODO add proper parameter
                                                # ===== collaborative planning with final AND intermediate auction =====
                                                # Note Test 01: 1 intermediate + 1 final with 50% of submitted requests each vs.
                                                #  1 final with 100% of submitted requests
                                                intermediate_auction = au.Auction(
                                                    tour_construction=tour_construction,
                                                    tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
                                                    request_selection=request_selection(
                                                        int(num_submitted_requests / 2)),  # TODO add proper parameter
                                                    bundle_generation=bundle_generation(
                                                        num_auction_bundles=num_auction_bundles / num_intermediate_auctions,
                                                        bundling_valuation=bundling_valuation(),
                                                        **bundle_generation_kwargs
                                                    ),
                                                    bidding=bd.ClearAndReinsertAll(
                                                        tour_construction,
                                                        tour_improvement(neighborhoods, tour_improvement_time_limit)
                                                    ),
                                                    winner_determination=wd.MaxBidGurobiCAP1(),
                                                    num_auction_rounds=num_final_auction_rounds  # TODO add proper parameter
                                                )

                                                final_auction = au.Auction(
                                                    tour_construction=tour_construction,
                                                    tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
                                                    request_selection=request_selection(
                                                        int(num_submitted_requests / 2)),  # TODO add proper parameter
                                                    bundle_generation=bundle_generation(
                                                        num_auction_bundles=num_auction_bundles / num_intermediate_auctions,
                                                        bundling_valuation=bundling_valuation(),
                                                        **bundle_generation_kwargs
                                                    ),
                                                    bidding=bd.ClearAndReinsertAll(
                                                        tour_construction,
                                                        tour_improvement(neighborhoods, tour_improvement_time_limit)
                                                    ),
                                                    winner_determination=wd.MaxBidGurobiCAP1(),
                                                    num_auction_rounds=num_final_auction_rounds
                                                )

                                                collaborative_planning_intermediate_final = dict(
                                                    time_window_offering=time_window_offering,
                                                    time_window_selection=time_window_selection,
                                                    tour_construction=tour_construction,
                                                    tour_improvement=tour_improvement(neighborhoods, tour_improvement_time_limit),
                                                    num_intermediate_auctions=num_intermediate_auctions,
                                                    intermediate_auction=intermediate_auction,
                                                    final_auction=final_auction,
                                                )

                                                yield collaborative_planning_intermediate_final
