from src.cr_ahd.auction_module.bidding import I1MarginalCostBidding
import numpy as np


def test_MarginalProfitBidding(bundle_set_a):
    carrier, bundle_set = bundle_set_a
    bids = []
    for i, bundle in bundle_set.items():
        mcb = I1MarginalCostBidding()._generate_bid(bundle, carrier)
        bids.append(mcb)
    test = np.round(bids, 13) == np.round(
        [30 + 10 * np.sqrt(5), 2 * np.sqrt(20 ** 2 + 20 ** 2), 30 + 10 * np.sqrt(5), ], 13)
    assert test.all()