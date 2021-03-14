from src.cr_ahd.utility_module import utils


def test_random_max_k_partition():
    """cannot actually be tested since it's random"""
    ls = range(10)
    k = 4
    partition = utils.random_max_k_partition(ls, k)
    assert len(partition) <= k
