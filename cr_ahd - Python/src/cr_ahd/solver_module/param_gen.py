from typing import List, Tuple, Dict

from src.cr_ahd.routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from src.cr_ahd.tw_management_module import tw_offering as two, tw_selection as tws


def parameter_generator():
    """
    generate dicts with all parameters are required to initialize a slv.Solver.
    """

    neighborhood_collections: List[List[nh.Neighborhood]] = [
        [
            nh.PDPMove(),
            nh.PDPTwoOpt(),
            nh.PDPRelocate()
        ],
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
        # mh.PDPTWSequentialLocalSearch,
        mh.PDPTWIteratedLocalSearch,
        mh.PDPTWVariableNeighborhoodDescent,
        # mh.PDPTWReducedVariableNeighborhoodSearch,
        mh.PDPTWVariableNeighborhoodSearch,
        mh.PDPTWSimulatedAnnealing,
        mh.NoMetaheuristic,
    ]

    tour_improvement_time_limit: List[int] = [
        1,
        5,
        15
    ]

    for tour_construction in tour_constructions:
        for tour_improvement in tour_improvements:
            for neighborhoods in neighborhood_collections:
                for time_window_offering, time_window_selection in [(two.FeasibleTW(), tws.UnequalPreference()),
                                                                    # (two.NoTw(), tws.NoTW())
                                                                    ]:
                    # ===== Isolated Planning Parameters, no auction =====
                    isolated_planning = dict(
                        time_window_offering=time_window_offering,
                        time_window_selection=time_window_selection,
                        tour_construction=tour_construction,
                        tour_improvement=tour_improvement(neighborhoods),
                        num_intermediate_auctions=0,
                        intermediate_auction=False,
                        final_auction=False,
                    )
                    yield isolated_planning
pass
