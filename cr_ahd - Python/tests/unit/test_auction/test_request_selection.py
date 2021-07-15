import src.cr_ahd.auction_module.request_selection as rs


def test_SingleLowestMarginalProfit(carrier_b):
    selected = rs.HighestInsertionCostDistance()._evaluate_requests(carrier_b, num_requests=5)
    assert [s.id_ for s in selected] == ['r2', 'r1', 'r3', 'r0', 'r4']


def test_Cluster(carrier_b):
    selected = rs.SpatialCluster()._evaluate_requests(carrier_b, 4)
    assert [s.id_ for s in selected] == ['r0', 'r1', 'r2', 'r3']
