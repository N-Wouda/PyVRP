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
    # If we start at time 'existing_time_warp' and finish at time 0, then the
    # time warp is 'existing_time_warp'.
    ds1 = DurationSegment(0, 0, 0, existing_time_warp, 0, 0)
    assert_equal(ds1.time_warp(), existing_time_warp)


def test_merge_two():
    """
    Tests merging two duration segments.
    """
    ds1 = DurationSegment(0, 0, 5, 0, 10, 0)
    ds2 = DurationSegment(1, 1, 0, 6, 1, 0)

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
    ds2 = DurationSegment(1, 1, 0, 6, 1, 3)
    merged = DurationSegment.merge(mat, ds1, ds2)
    assert_equal(merged.time_warp(), 11)


def test_merge_two_with_time_warp_and_wait_time():
    """
    Tests merging two duration segments, where both have time warp so no slack.
    However, when merging the two segments, we have a waiting time in between.
    """
    ds1 = DurationSegment(0, 0, 5, 1, 2, 0)
    ds2 = DurationSegment(1, 1, 5, 20, 21, 0)

    mat = np.asarray([[0, 4], [1, 0]])
    merged = DurationSegment.merge(mat, ds1, ds2)

    # There is no release time. The first segment (ds1) starts at 1, takes 5
    # and there is 4 time warp so finishes at 1 + 5 - 4 = 2. The second segment
    # starts at 20, so we incur 18 waiting time. It then takes 5, and has 4
    # time warp so finishes at 20 + 5 - 4 = 21.

    # The total duration should be 5 + 5 + 18 = 28, and the total time warp
    # should be 8, such that indeed the latest finish is at 1 + 28 - 8 = 21.
    assert_equal(merged.time_warp(), 8)
    assert_equal(merged.duration(), 28)


def test_merge_three():
    """
    Tests merging three duration segments.
    """
    ds1 = DurationSegment(0, 0, 5, 0, 10, 0)
    ds2 = DurationSegment(1, 1, 0, 3, 6, 0)
    ds3 = DurationSegment(2, 2, 0, 2, 3, 2)

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
    ds1 = DurationSegment(0, 0, 5, 0, 5 - time_warp, 0)  # depot
    ds2 = DurationSegment(1, 1, 1, 3, 4 - time_warp, 0)  # client 1

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
    # t = 8, which is five units after the time window closes. This adds 5
    # time warp to the segment, which, together with the initial time warps
    # makes for 5 + 1 + 1 = 7 total time warp.
    assert_equal(merged12.time_warp(), 7)

    # We can leave at the earliest at t = 3, the start of the client's time
    # window. We then arrive at t = 6, which adds six units of time warp.
    # Combined with the initial time warp, this results in 6 + 1 + 1 = 8 units
    # of total time warp.
    assert_equal(merged21.time_warp(), 8)

    # This merged DS represents the route plan 0 -> 1 -> 1 -> 0. We leave 1
    # at t = 10 - 7, and travel takes no time. We do get the 1 unit of
    # existing time warp. We then travel to the depot, arriving at t = 6. This
    # is 6 time units after the time window closes, which adds 6 time warp
    # (plus the existing unit). So we get 7 + 1 + 1 + 6 = 15 time warp.
    merged = DurationSegment.merge(mat, merged12, merged21)
    assert_equal(merged.time_warp(), 15)


def test_max_duration_argument():
    """
    Tests that the ``max_duration`` argument is evaluated correctly, and indeed
    increases the time warp if it's violated.
    """
    ds = DurationSegment(0, 0, 5, 0, 5, 0)  # five duration (and latest finish)

    assert_equal(ds.time_warp(), 0)  # default not duration limited
    assert_equal(ds.time_warp(max_duration=2), 3)
    assert_equal(ds.time_warp(max_duration=0), 5)


def test_OkSmall_with_time_warp(ok_small):
    """
    Tests a small example route using the OkSmall instance. In particular, we
    also check that duration does not include time warp.
    """
    segments = [
        DurationSegment(
            idx_first=idx,
            idx_last=idx,
            duration=loc.service_duration if idx != 0 else 0,
            earliest_start=loc.tw_early,
            latest_finish=(
                loc.tw_late + (loc.service_duration if idx != 0 else 0)
            ),
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
    that has ``latestFinish = INT_MAX`` results in integer overflow.
    """
    matrix = np.array([[0, 0], [0, 0]])
    ds1 = DurationSegment(1, 1, 9, 18, 9, 0)

    assert_(ds1.duration() < ds1.time_warp())

    ds2 = DurationSegment(0, 0, 0, 0, np.iinfo(np.int64).max, 0)
    assert_equal(ds2.latest_start(), np.iinfo(np.int64).max)

    # ds1 has 9 duration, 18 earliest start and 9 latest finish so 9 + 18 - 9 =
    # 18 time warp, which results in an arrival time of
    # -9 at ds2. Before enforcing non-negative arrival times, this would result
    # in an integer overflow when subtracting this arrival time from ds2's
    # tw_late.
    ds = DurationSegment.merge(matrix, ds1, ds2)
    assert_equal(ds.time_warp(), 18)
