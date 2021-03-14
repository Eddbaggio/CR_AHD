from pathlib import Path

from src.cr_ahd.core_module.instance import read_custom_json_instance, read_solomon


def test_read_custom():
    inst = read_custom_json_instance(Path(
        'C:/Users/Elting/ucloud/PhD/02_Research/02_Collaborative Routing for Attended Home Deliveries/01_Code/cr_ahd - Python/tests/fixtures/C101_3_15_ass_#014.json'))
    assert inst.num_requests == 100
    assert len(inst.unrouted_requests) == 100
    assert inst.num_carriers == 3
    assert inst.num_vehicles == 45
    pass


def test_read_solomon():  # TODO: why is this sooo slow in debug mode?
    inst = read_solomon('R209', 1)
    assert inst.num_requests == 100
    assert len(list(inst.unrouted_requests)) == 100
    assert inst.num_carriers == 1
    pass
