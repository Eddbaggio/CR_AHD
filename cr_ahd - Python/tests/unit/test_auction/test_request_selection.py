import src.cr_ahd.auction_module.request_selection as rs


def test_SingleLowestMarginalProfit(carrier_b):
    selected = rs.FiftyPercentHighestMarginalCost()._select_requests(carrier_b, n=5)
    assert [s.id_ for s in selected] == ['r2', 'r1', 'r3', 'r0', 'r4']
