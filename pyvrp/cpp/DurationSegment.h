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
 *     time_warp: int,
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
 *     Earliest visit moment of the first client.
 * latest_finish
 *    Latest finish moment of the last client.
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
    [[nodiscard]] Duration latestStart() const;

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
                           Duration timeWarp,
                           Duration earliestStart,
                           Duration latestStart,
                           Duration releaseTime);

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

    // We compute the new latest finish time for the merged segment forward
    // from the latest finish time of the first segment.
    // Formally, we should also include timeWarp_ in the second segment in the
    // forward computation, however:
    // - If other.timeWarp_ = 0, we can omit it from the computation
    // - If other.timeWarp_ > 0, then other.earliestStart_ + other.duration_ -
    //   other.timeWarp_ = other.latestFinish_
    //   - If waitDuration > 0, it holds that:
    //     latestFinish_ + edgeDuration + waitDuration + other.duration_ -
    //     other.timeWarp_
    //     = other.earliestStart_ + other.duration_ - other.timeWarp_
    //     = other.latestFinish_
    //   - If waitDuration = 0, it holds that:
    //     latestFinish_ + edgeDuration >= other.earliestStart_
    //     so latestFinish_ + edgeDuration + other.duration_ - other.timeWarp_
    //     >= other.earliestStart_ + other.duration_ - other.timeWarp_
    //     = other.latestFinish_
    //  so in both cases the maximum is equal to latestFinish_, and this remains
    //  true if we ignore timeWarp_ in the computation as that will only
    //  increase the left operand of the min.

    // We compute the new earliest start time for the merged segment backwards
    // from the earliest start time for the second segment.
    // Formally, we should also include timeWarp_ in the first segment in the
    // backwards computation, however:
    // - If timeWarp = 0, we can omit it from the computation
    // - If timeWarp > 0, then earliestStart_ + duration_ - timeWarp_ =
    // latestFinish_
    //   - If waitDuration > 0, it holds that:
    //     other.earliestStart_ - waitDuration - edgeDuration - (duration_ -
    //     timeWarp_)
    //     = latestFinish_ - (duration_ - timeWarp_) = earliestStart_
    //   - If waitDuration = 0, it holds that:
    //     latestFinish_ >= other.earliestStart_ - edgeDuration
    //     so earliestStart_ + duration_ - timeWarp_ >= other.earliestStart_
    //     - edgeDuration
    //     so earliestStart_ >= other.earliestStart_ - edgeDuration - duration_
    //     + timeWarp_
    //  so in both cases the maximum is equal to earliestStart_, and this
    //  remains true if we ignore timeWarp_ in the computation as that will only
    //  decrease the left operand of the max.

    Dur const diffDuration = edgeDuration + waitDuration + other.duration_;
    Dur const mergedLatestFinish
        = latestFinish_ < other.latestFinish_ - diffDuration
              ? latestFinish_ + diffDuration
              : other.latestFinish_;  // Avoid overflow
    return {
        idxFirst_,
        other.idxLast_,
        duration_ + diffDuration,
        std::max(other.earliestStart_ - waitDuration - edgeDuration - duration_,
                 earliestStart_),
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
    Duration const latestStart = std::max<Duration>(latestFinish_ - duration_, earliestStart_);
    return std::max<Duration>(earliestStart_ + duration_ - latestFinish_, 0)
         + std::max<Duration>(releaseTime_ - latestStart, 0)
         + std::max<Duration>(duration_ - maxDuration, 0);
    // clang-format on
}

DurationSegment::DurationSegment(size_t idxFirst,
                                 size_t idxLast,
                                 Duration duration,
                                 Duration timeWarp,
                                 Duration earliestStart,
                                 Duration latestStart,
                                 Duration releaseTime)
    : idxFirst_(idxFirst),
      idxLast_(idxLast),
      duration_(duration),
      earliestStart_(earliestStart),
      latestFinish_(latestStart + duration - timeWarp),
      releaseTime_(releaseTime)
{
    assert(timeWarp == 0 || earliestStart == latestStart);
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
