import pytest

import src.cr_ahd.tw_management_module.tw_offering as two


class TestTWOfferingBehavior:
    @pytest.mark.parametrize('tour_length, e, l, expected_fi, expected_li',
                             [
                                 pytest.param(7, 4, 6, 1, 3),
                                 pytest.param(7, 0, 7.5, 1, 4),
                                 pytest.param(7, 11.5, 14.5, 5, 8),
                                 pytest.param(7, 10, 12, 4, 6),
                                 pytest.param(0, 3, 6, 1, 1),
                                 pytest.param(3, 0, 4, 1, 2),
                             ])
    def test_find_insertion_index(self, tw_hour, spiral_tour, tour_length, e, l, expected_fi, expected_li):
        tw = tw_hour(e, l)
        tour = spiral_tour(tour_length)
        fi = two.FeasibleTW()._find_first_insertion_index(tw, tour)
        assert fi == expected_fi
        li = two.FeasibleTW()._find_last_insertion_index(tw, tour, fi)
        assert li == expected_li

#
# class TestFeasibleTW:
#     def test__evaluate_time_window(self, tw_0_2, carrier_a_partially_routed, ):
#         ev_0 = two.FeasibleTW()._evaluate_time_window(tw_0_2, carrier_a_partially_routed,
#                                                       carrier_a_partially_routed.requests[-1])
#         # TODO need to test the edge cases
#         ev_1 = two.FeasibleTW()._evaluate_time_window(tw_0_2, carrier_a_partially_routed,
#                                                       carrier_a_partially_routed.requests[-1])
#         self.fail()
