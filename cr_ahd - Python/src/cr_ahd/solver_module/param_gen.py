from typing import List, Tuple, Dict

from src.cr_ahd.auction_module import request_selection as rs, bundle_generation as bg, bundle_valuation as bv, \
    auction as au, bidding as bd, winner_determination as wd
from src.cr_ahd.routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from src.cr_ahd.tw_management_module import tw_management as twm, tw_offering as two, tw_selection as tws


def parameter_generator():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """
    neighborhoods: List[nh.Neighborhood] = [  # these are fixed at the moment, i.e. not looped over
        nh.PDPMove(),
        nh.PDPTwoOpt(),
        # nh.PDPRelocate()
    ]

    tour_constructions: List[cns.PDPParallelInsertionConstruction] = [
        cns.MinTravelDistanceInsertion(),
        # cns.MinTimeShiftInsertion()
    ]

    tour_improvements: List = [
        # mh.LocalSearchFirst([neighborhoods[0]]),
        # mh.LocalSearchFirst([neighborhoods[1]]),
        # mh.LocalSearchBest([neighborhoods[0]]),
        # mh.LocalSearchBest([neighborhoods[1]]),
        # mh.PDPTWSequentialLocalSearch(neighborhoods),
        # mh.PDPTWIteratedLocalSearch(neighborhoods),
        mh.PDPTWVariableNeighborhoodDescent(neighborhoods),
        # mh.PDPTWReducedVariableNeighborhoodSearch(neighborhoods),
        # mh.PDPTWSimulatedAnnealing(neighborhoods),
        # mh.NoMetaheuristic([]),
    ]

    time_window_managements: List[twm.TWManagement] = [
        twm.TWManagementSingle(two.FeasibleTW(),
                               tws.UnequalPreference()),
        # twm.TWManagementNoTW(None, None)
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

    for tour_construction in tour_constructions:
        for tour_improvement in tour_improvements:
            for time_window_management in time_window_managements:
                # Isolated Planning Parameters, no auction
                yield dict(tour_construction=tour_construction,
                           tour_improvement=tour_improvement,
                           time_window_management=time_window_management,
                           auction=False,
                           )
                for num_submitted_requests in nums_submitted_requests:
                    for request_selection in request_selections:
                        for num_auction_bundles in nums_auction_bundles:
                            for bundle_generation, bundle_generation_kwargs in bundle_generations:
                                for bundling_valuation in bundling_valuations:
                                    # auction for collaborative planning
                                    auction = au.Auction(tour_construction,
                                                         tour_improvement,
                                                         request_selection(num_submitted_requests),
                                                         bundle_generation(num_auction_bundles=num_auction_bundles,
                                                                           bundling_valuation=bundling_valuation(),
                                                                           **bundle_generation_kwargs
                                                                           ),
                                                         bidding=bd.DynamicInsertionAndImprove(tour_construction,
                                                                                               tour_improvement),
                                                         winner_determination=wd.MaxBidGurobiCAP1(),
                                                         )
                                    # collaborative planning
                                    yield dict(tour_construction=tour_construction,
                                               tour_improvement=tour_improvement,
                                               time_window_management=time_window_management,
                                               auction=auction,
                                               )
    pass