#ifndef PYVRP_DURATIONSEGMENT_H
#define PYVRP_DURATIONSEGMENT_H

#include "Matrix.h"
#include "Measure.h"
#include "ProblemData.h"

namespace pyvrp
{
/**
 * DurationSegment(
 *     idx_first: int,
 *     idx_last: int,
 *     duration: int,
 *     earliest_start: int,
 *     latest_finish: int,
 *     release_time: int,
 * )
 *
 * Creates a duration segment.
 *
 * Duration segments can be efficiently concatenated, and track statistics
 * about route duration and time warp resulting from visiting clients in the
 * concatenated order.
 *
 * Parameters
 * ----------
 * idx_first
 *     Index of the first client in the route segment.
 * idx_last
 *     Index of the last client in the route segment.
 * duration
 *     Total duration, including waiting time.
 * earliest_start
 *     Earliest moment to start service at the first client in the segment that
 *     will result in minimal time warp, waiting time and route duration.
 * latest_finish
 *     Latest moment to finish service of the last client in the segment that
 *     will result in minimal time warp, waiting time and route duration.
 * release_time
 *     Earliest moment to start the route segment.
 */
class DurationSegment
{
    size_t idxFirst_;         // Index of the first client in the segment
    size_t idxLast_;          // Index of the last client in the segment
    Duration duration_;       // Total duration, incl. waiting and servicing
    Duration earliestStart_;  // Earliest visit moment of first client
    Duration latestFinish_;   // Latest finish moment of last client
    Duration releaseTime_;    // Earliest allowed moment to leave the depot

    [[nodiscard]] inline DurationSegment
    merge(Matrix<Duration> const &durationMatrix,
          DurationSegment const &other) const;

public:
    template <typename... Args>
    [[nodiscard]] static DurationSegment
    merge(Matrix<Duration> const &durationMatrix,
          DurationSegment const &first,
          DurationSegment const &second,
          Args &&...args);

    /**
     * The total duration of this route segment.
     */
    [[nodiscard]] inline Duration duration() const;

    /**
     * Returns the time warp on this route segment. Additionally, any time warp
     * incurred by violating the maximum duration argument is also counted.
     *
     * Parameters
     * ----------
     * max_duration
     *     Maximum allowed duration, if provided. If the segment's duration
     *     exceeds this value, any excess duration is counted as time warp.
     *     Default unconstrained.
     *
     * Returns
     * -------
     * int
     *     Total time warp on this route segment.
     */
    [[nodiscard]] inline Duration
    timeWarp(Duration const maxDuration
             = std::numeric_limits<Duration>::max()) const;

    /**
     * Earliest start time for this route segment that results in minimum route
     * segment duration.
     */
    [[nodiscard]] Duration earliestStart() const;

    /**
     * Latest start time for this route segment that results in minimum route
     * segment duration.
     */
    [[nodiscard]] inline Duration latestStart() const;

    /**
     * Earliest possible release time of the clients in this route segment.
     */
    [[nodiscard]] Duration releaseTime() const;

    // Construct from attributes of the given client.
    DurationSegment(size_t idx, ProblemData::Client const &client);

    // Construct from raw data.
    inline DurationSegment(size_t idxFirst,
                           size_t idxLast,
                           Duration duration,
                           Duration earliestStart,
                           Duration latestFinish,
                           Duration releaseTime);

    // Move or copy construct from the other duration segment.
    inline DurationSegment(DurationSegment const &) = default;
    inline DurationSegment(DurationSegment &&) = default;

    // Move or copy assign form the other duration segment.
    inline DurationSegment &operator=(DurationSegment const &) = default;
    inline DurationSegment &operator=(DurationSegment &&) = default;
};

DurationSegment DurationSegment::merge(Matrix<Duration> const &durationMatrix,
                                       DurationSegment const &other) const
{
    using Dur = pyvrp::Duration;

    // edgeDuration is the travel duration from our last to the other's first
    // client
    Dur const edgeDuration = durationMatrix(idxLast_, other.idxFirst_);

    // We must wait if we arrive before the earliest start time of the other
    // segment. Note that the earliest start time of the other segment may not
    // be the opening of the time window of the first client, but rather the
    // time we should start to avoid unnecessary waiting time. We may still
    // start earlier at that other segment, but this suggests that we should
    // wait at another moment for the same total duration since we cannot
    // finish the other route segment earlier.
    Dur const waitDuration
        = latestFinish_ < other.earliestStart_ - edgeDuration
              ? other.earliestStart_ - latestFinish_ - edgeDuration
              : 0;  // ternary rather than max avoids underflow

    // The new earliest start time for the merged segment is either the
    // earliest start time for the first/current segment or a later start time
    // computed backwards from the earliest start time of the second/other
    // segment. In this computation, we can ignore existing time warp in the
    // first segment, since if there is existing time warp, there is no slack
    // in the first segment and the earliest start time will not change.
    Dur const mergedEarliestStart = std::max(other.earliestStart_ - waitDuration
                                                 - edgeDuration - duration_,
                                             earliestStart_);

    // The new latest finish time for the merged segment is either the latest
    // finish time for the second/other segment or an earlier finish time
    // computed forward from the latest finish time of the first/current
    // segment. In this computation, we can ignore existing time warp in the
    // second segment, since if there is existing time warp, there is no slack
    // in the second segment and the latest finish time will not change.
    Dur const diffDuration = edgeDuration + waitDuration + other.duration_;
    Dur const mergedLatestFinish
        = latestFinish_ < other.latestFinish_ - diffDuration
              ? latestFinish_ + diffDuration
              : other.latestFinish_;  // Avoid overflow

    return {idxFirst_,
            other.idxLast_,
            duration_ + diffDuration,
            mergedEarliestStart,
            mergedLatestFinish,
            std::max(releaseTime_, other.releaseTime_)};
}

template <typename... Args>
DurationSegment
DurationSegment::merge([[maybe_unused]] Matrix<Duration> const &durationMatrix,
                       [[maybe_unused]] DurationSegment const &first,
                       [[maybe_unused]] DurationSegment const &second,
                       [[maybe_unused]] Args &&...args)
{
#ifdef PYVRP_NO_TIME_WINDOWS
    return {0, 0, 0, 0, 0, 0, 0};
#else
    auto const res = first.merge(durationMatrix, second);

    if constexpr (sizeof...(args) == 0)
        return res;
    else
        return merge(durationMatrix, res, args...);
#endif
}

Duration DurationSegment::duration() const { return duration_; }

Duration DurationSegment::timeWarp(Duration const maxDuration) const
{
    // clang-format off
    return std::max<Duration>(earliestStart_ + duration_ - latestFinish_, 0)
         + std::max<Duration>(releaseTime_ - latestStart(), 0)
         + std::max<Duration>(duration_ - maxDuration, 0);
    // clang-format on
}

Duration DurationSegment::latestStart() const
{
    return std::max<Duration>(latestFinish_ - duration_, earliestStart_);
}

DurationSegment::DurationSegment(size_t idxFirst,
                                 size_t idxLast,
                                 Duration duration,
                                 Duration earliestStart,
                                 Duration latestFinish,
                                 Duration releaseTime)
    : idxFirst_(idxFirst),
      idxLast_(idxLast),
      duration_(duration),
      earliestStart_(earliestStart),
      latestFinish_(latestFinish),
      releaseTime_(releaseTime)
{
}
}  // namespace pyvrp

#endif  // PYVRP_DURATIONSEGMENT_H
