from src.cr_ahd.auction_module.winner_determination import LowestBid


# bids: Dict[List[Vertex], Dict[Carrier, float]]
def test_lowest_bid_winner_determination(bids_a):
    LowestBid().execute(bids_a)
    assignments = []
    for bundle, _ in bids_a.items():
        for request in bundle:
            assignments.append(request.carrier_assignment)
    assert assignments == ['c2', 'c2', 'c0', 'c0', 'c1', 'c1']
