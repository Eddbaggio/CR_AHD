import datetime as dt

import numpy as np
import pytest

import src.cr_ahd.utility_module.utils as ut
import src.cr_ahd.auction_module.bidding as bd


def test_I1TravelDurationIncrease(bundle_set_a):
    carrier, bundle_set = bundle_set_a
    bids = []
    for i, bundle in bundle_set.items():
        mcb = bd.I1TravelDurationIncrease()._value_with_bundle(bundle, carrier, [], )
        bids.append(mcb)
    for i, dist in enumerate([30 + 10 * np.sqrt(5), 2 * np.sqrt(20 ** 2 + 20 ** 2), 30 + 10 * np.sqrt(5), ]):
        assert abs(ut.travel_time(dist) - bids[i]) < dt.timedelta(seconds=1)


def test_I1TravelDistanceIncrease(bundle_set_a):
    carrier, bundle_set = bundle_set_a
    bids = []
    for i, bundle in bundle_set.items():
        mcb = bd.I1TravelDistanceIncrease()._value_with_bundle(bundle, carrier, [], )
        bids.append(mcb)
    assert bids == pytest.approx([30 + 10 * np.sqrt(5), 2 * np.sqrt(20 ** 2 + 20 ** 2), 30 + 10 * np.sqrt(5), ])
