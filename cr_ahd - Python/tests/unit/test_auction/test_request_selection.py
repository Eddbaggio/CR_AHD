import src.cr_ahd.auction_module.request_selection as rs


def test_SingleLowestMarginalProfit(carrier_b):
    selected = rs.SingleLowestMarginalCost()._select_requests(carrier_b)
    assert selected[0].id_ == 'r4'
