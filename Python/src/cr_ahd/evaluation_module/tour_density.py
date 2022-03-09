from typing import Sequence
import datetime as dt
from core_module import tour as tr
from tw_management_module.tw import TimeWindow


def tour_density(tour: tr.Tour, time_windows: Sequence[TimeWindow]):
    tw_occupancy = {tw: {'travel': dt.timedelta(0), 'service': dt.timedelta(0), 'wait': dt.timedelta(0)}
                    for tw in time_windows}

    # select the initial tw
    tw_index = 0
    tw = time_windows[tw_index]

    t = tour.arrival_time_sequence[0]  # start of the tour
    for j_pos in range(1, len(tour)):

        i_departure = tour.service_time_sequence[j_pos - 1] + tour.service_duration_sequence[j_pos - 1]
        j_arrival = tour.arrival_time_sequence[j_pos]
        travel_delta = j_arrival - i_departure
        t += travel_delta
        if t <= tw.close:
            tw_occupancy[tw]['travel'] += travel_delta
        else:
            tw_occupancy[tw]['travel'] += tw.close - i_departure
            # switch to the next time window
            tw_index += 1
            tw = time_windows[tw_index]
            tw_occupancy[tw]['travel'] += t - tw.open

        # wait
        wait_delta = tour.wait_duration_sequence[j_pos]
        t += wait_delta
        if t <= tw.close:
            tw_occupancy[tw]['wait'] += wait_delta
        else:
            tw_occupancy[tw]['wait'] += tw.close - j_arrival
            # switch to the next time window
            tw_index += 1
            tw = time_windows[tw_index]
            tw_occupancy[tw]['wait'] += t - tw.open

        # service
        service_delta = tour.service_duration_sequence[j_pos]
        t += service_delta
        if t <= tw.close:
            tw_occupancy[tw]['service'] += service_delta
        else:
            tw_occupancy[tw]['service'] += tw.close - tour.service_time_sequence[j_pos]
            # switch to the next time window
            tw_index += 1
            tw = time_windows[tw_index]
            tw_occupancy[tw]['service'] += t - tw.open

    # print
    for tw, occupancy in tw_occupancy.items():
        print(tw)
        print(str(occupancy))
        sum_durations = sum(occupancy.values(), dt.timedelta(0))
        assert sum_durations <= tw.duration
        print(sum_durations)
    pass
