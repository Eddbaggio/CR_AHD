def test_retract_unrouted(carrier_b):
    carrier_b.retract_requests_and_update_routes(carrier_b.requests)
    assert bool(carrier_b.requests) is False
