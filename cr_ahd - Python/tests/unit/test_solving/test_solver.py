import src.cr_ahd.solving_module.solver as sl


def test_DynamicI1InsertionWithAuctionSolver(instance_a):
    # TODO instance_a has randomly located vertices --> cannot be tested!
    sl.DynamicI1InsertionWithAuctionSolver().execute(instance_a)
    pass