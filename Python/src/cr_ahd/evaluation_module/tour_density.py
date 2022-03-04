from typing import Sequence
import datetime as dt
from core_module import tour as tr
from tw_management_module.tw import TimeWindow


def tour_density(tour: tr.Tour, time_windows: Sequence[TimeWindow]):
    tw_time_split = {tw: {'travel': dt.timedelta(0), 'service': dt.timedelta(0), 'wait': dt.timedelta(0)}
                     for tw in time_windows}
    for j in range(1, len(tour)):
        i_arrival = tour.arrival_time_sequence[j-1]
        i_wait = tour.wait_duration_sequence[j-1]
        i_service = tour.service_time_sequence[j-1]
        j_arrival = tour.arrival_time_sequence[j]
        j_wait = tour.wait_duration_sequence[j]
        j_service = tour.service_time_sequence[j]
        travel_duration = j_service - j_wait

