import src.cr_ahd.tw_management_module.tw_selection as tws
import src.cr_ahd.utility_module.utils as ut


class TestEarlyPreference:
    def test_select_tw(self, tw_offer_set):
        selected_tw = tws.EarlyPreference().execute(tw_offer_set(10, ut.TW_LENGTH, True, True))
        assert selected_tw == ut.TimeWindow(ut.START_TIME, ut.START_TIME + ut.TW_LENGTH)
