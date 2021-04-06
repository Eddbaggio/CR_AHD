import src.cr_ahd.solver as sl


def test_DynamicI1InsertionWithAuctionSolver(instance_a):
    # TODO instance_a has randomly located vertices --> cannot be tested!
    sl.DynamicI1InsertionWithAuctionA().execute(instance_a)
    pass
