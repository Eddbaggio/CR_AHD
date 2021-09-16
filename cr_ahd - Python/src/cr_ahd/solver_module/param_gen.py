from typing import List, Tuple, Dict

from src.cr_ahd.routing_module import neighborhoods as nh, tour_construction as cns, metaheuristics as mh
from src.cr_ahd.tw_management_module import tw_offering as two, tw_selection as tws


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
        mh.PDPTWIteratedLocalSearch,
        mh.PDPTWVariableNeighborhoodDescent,
        # mh.PDPTWReducedVariableNeighborhoodSearch,
        mh.PDPTWVariableNeighborhoodSearch,
        mh.PDPTWSimulatedAnnealing,
        mh.NoMetaheuristic,
    ]

    neighborhood_collections: List[List[nh.Neighborhood]] = [
        [nh.PDPMove(), nh.PDPTwoOpt(), nh.PDPRelocate()],
    ]
    tour_improvement_time_limits: List[float] = [
        1,
        # 2,
        # 5,
        10
    ]

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
