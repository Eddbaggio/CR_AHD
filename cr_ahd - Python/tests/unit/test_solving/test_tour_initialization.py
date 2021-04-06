from src.cr_ahd.routing_module.tour_initialization import EarliestDueDate


class TestEarliestDueDate:
    def test_find_seed_request(self, carrier_c):
        seed = EarliestDueDate().find_seed_request(carrier_c)
        assert seed.id_ == 'r3'
