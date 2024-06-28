import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from pyvrp._pyvrp import DurationSegment


@pytest.mark.parametrize("existing_time_warp", [2, 5, 10])
def test_time_warp_when_there_is_existing_time_warp(existing_time_warp):
    """
    Tests that the ``time_warp()`` returns existing time warp when no
    segments have been merged yet.
    """
    ds1 = DurationSegment(0, 0, 0, existing_time_warp, 0, 0, 0)
    assert_equal(ds1.time_warp(), existing_time_warp)


def test_merge_two():
    """
    Tests merging two duration segments.
    """
    ds1 = DurationSegment(0, 0, 5, 0, 0, 5, 0)
    ds2 = DurationSegment(1, 1, 0, 5, 3, 6, 0)

    mat = np.asarray([[1, 4], [1, 2]])
    merged = DurationSegment.merge(mat, ds1, ds2)

    # There is no release time. The first stop (ds1) takes already has five
    # duration, and starts at time 0. Then, we have to drive for 4 units
    # (mat(0, 1) = 4) to get to the second stop (ds2). This second segment has
    # 5 time warp, and we arrive there at time 5 + 4 = 9, which is 9 - 6 = 3
    # after its closing time window. So we get a final time warp of 5 + 3 = 8.
    assert_equal(merged.time_warp(), 8)

    # Now, let's add a bit of release time (3), so that the total time warp
    # should become 8 + 3 = 11.
    ds2 = DurationSegment(1, 1, 0, 5, 3, 6, 3)
    merged = DurationSegment.merge(mat, ds1, ds2)
    assert_equal(merged.time_warp(), 11)


def test_merge_three():
    """
    Tests merging three duration segments.
    """
    ds1 = DurationSegment(0, 0, 5, 0, 0, 5, 0)
    ds2 = DurationSegment(1, 1, 0, 0, 3, 6, 0)
    ds3 = DurationSegment(2, 2, 0, 0, 2, 3, 2)

    mat = np.asarray([[1, 4, 1], [1, 2, 4], [1, 1, 1]])
    merged1 = DurationSegment.merge(mat, ds1, ds2)
    merged2 = DurationSegment.merge(mat, merged1, ds3)
    merged3 = DurationSegment.merge(mat, ds1, ds2, ds3)

    # Merge all together should be the same as merging in several steps.
    assert_equal(merged3.time_warp(), merged2.time_warp())

    # After also merging in ds3, we should have 3 time warp from 0 -> 1, and 7
    # time warp from 1 -> 2. Since there's also a release time of 2, the total
    # time warp is 12.
    assert_equal(merged3.time_warp(), 12)


def test_merging_two_previously_merged_duration_segments():
    """
    This test evaluates what happens when we merge two previously merged
    segments, when both have time warp.
    """
    time_warp = 1
    ds1 = DurationSegment(0, 0, 5, time_warp, 0, 5, 0)  # depot
    ds2 = DurationSegment(1, 1, 1, time_warp, 3, 6, 0)  # client 1

    # Each of these segments has some initial time warp.
    assert_equal(ds1.time_warp(), 1)
    assert_equal(ds2.time_warp(), 1)

    mat = np.asarray([[0, 4], [3, 0]])
    merged12 = DurationSegment.merge(mat, ds1, ds2)  # depot -> client
    merged21 = DurationSegment.merge(mat, ds2, ds1)  # client -> depot

    # The order in which duration segments are merged matters, so although
    # these two segments cover the same clients, the total time warp is not
    # the same.
    assert_(merged12.time_warp() != merged21.time_warp())

    # Both segments start with 1 initial time warp. Going from depot -> client
    # happens at t = 5 - 1, and takes 4 time units to complete. So we arrive at
    # t = 8, which is three units after the time window closes. This adds 2
    # time warp to the segment, which, together with the initial time warps
    # makes for 2 + 1 + 1 = 4 total time warp.
    assert_equal(merged12.time_warp(), 4)

    # We can leave at the earliest at t = 3, the start of the client's time
    # window. We then arrive at t = 6, which adds one unit of time warp.
    # Combined with the initial time warp, this results in 1 + 1 + 1 = 3 units
    # of total time warp.
    assert_equal(merged21.time_warp(), 3)

    # This merged DS represents the route plan 0 -> 1 -> 1 -> 0. We leave 1
    # at t = 10 - 4, and travel takes no time. We do get the 1 unit of
    # existing time warp. We then travel to the depot, arriving at t = 9. This
    # is 4 time units after the time window closes, which adds 4 time warp
    # (plus the existing unit). So we get 4 + 1 + 1 + 4 = 10 time warp.
    merged = DurationSegment.merge(mat, merged12, merged21)
    assert_equal(merged.time_warp(), 10)


def test_max_duration_argument():
    """
    Tests that the ``max_duration`` argument is evaluated correctly, and indeed
    increases the time warp if it's violated.
    """
    ds = DurationSegment(0, 0, 5, 0, 0, 0, 0)  # five duration

    assert_equal(ds.time_warp(), 0)  # default not duration limited
    assert_equal(ds.time_warp(max_duration=2), 3)
    assert_equal(ds.time_warp(max_duration=0), 5)


def test_OkSmall_with_time_warp(ok_small):
    """
    Tests a small example route using the OkSmall instance. In particular, we
    also check that duration does not include time warp.
    """
    vehicle_type = ok_small.vehicle_type(0)
    segments = [
        DurationSegment(
            idx_first=idx,
            idx_last=idx,
            duration=loc.service_duration if idx != 0 else 0,
            time_warp=0,
            tw_early=loc.tw_early if idx > 0 else vehicle_type.tw_early,
            tw_late=loc.tw_late if idx > 0 else vehicle_type.tw_late,
            release_time=loc.release_time if idx != 0 else 0,
        )
        for idx, loc in enumerate(ok_small.depots() + ok_small.clients())
    ]

    # Create the DS associated with route 0 -> 1 -> 3 -> 0 (so depot to 1,
    # to 3, and back to depot).
    ds = segments[0]
    for idx in [1, 3, 0]:
        mat = ok_small.duration_matrix(profile=0)
        ds = DurationSegment.merge(mat, ds, segments[idx])

    # First the route's duration. This depends on the travel duration, service
    # time, and possible waiting time. We do not have waiting time on this
    # route. So all we need to determine is:
    #   Travel durations:
    #       - 0 -> 1: 1544
    #       - 1 -> 3: 1427
    #       - 3 -> 0: 2063
    #   Service times:
    #       - 1: 360
    #       - 3: 420
    assert_equal(ds.duration(), 1544 + 1427 + 2063 + 360 + 420)

    # But there is time warp as well, because 1's time window opens at 15600,
    # while 3's time window closes at 15300. So we leave 1 at 15600 + 360, then
    # drive 1427 and arrive at 3 at 15600 + 360 + 1427 = 17387. We then warp
    # back in time to 15300, for 17387 - 15300 = 2087 time warp.
    assert_equal(ds.time_warp(), 2087)


def test_bug_fix_overflow_more_timewarp_than_duration():
    """
    This test exercises the issue identified in #588, when merging a duration
    segment that has more time warp than duration with another duration segment
    that has ``twLate = INT_MAX`` results in integer overflow.
    """
    matrix = np.array([[0, 0], [0, 0]])

    ds1 = DurationSegment(1, 1, 9, 18, 0, 18, 0)
    assert_(ds1.duration() < ds1.time_warp())

    ds2 = DurationSegment(0, 0, 0, 0, 0, np.iinfo(np.int64).max, 0)
    assert_equal(ds2.tw_late(), np.iinfo(np.int64).max)

    # ds1 has 9 duration and 18 time warp, which results in an arrival time of
    # -9 at ds2. Before enforcing non-negative arrival times, this would result
    # in an integer overflow when subtracting this arrival time from ds2's
    # tw_late.
    ds = DurationSegment.merge(matrix, ds1, ds2)
    assert_equal(ds.time_warp(), 18)
