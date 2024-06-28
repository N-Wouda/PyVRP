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
 *     tw_early: int,
 *     tw_late: int,
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
 * time_warp
 *     Total time warp on the route segment.
 * tw_early
 *     Earliest visit moment of the first client.
 * tw_late
 *     Latest visit moment of the first client.
 * release_time
 *     Earliest moment to start the route segment.
 */
class DurationSegment
{
    size_t idxFirst_;       // Index of the first client in the segment
    size_t idxLast_;        // Index of the last client in the segment
    Duration duration_;     // Total duration, incl. waiting and servicing
    Duration timeWarp_;     // Cumulative time warp
    Duration twEarly_;      // Earliest visit moment of first client
    Duration twLate_;       // Latest visit moment of first client
    Duration releaseTime_;  // Earliest allowed moment to leave the depot

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
    [[nodiscard]] Duration twEarly() const;

    /**
     * Latest start time for this route segment that results in minimum route
     * segment duration.
     */
    [[nodiscard]] Duration twLate() const;

    /**
     * Earliest possible release time of the clients in this route segment.
     */
    [[nodiscard]] Duration releaseTime() const;

    // Construct from attributes of the given client.
    DurationSegment(size_t idx, ProblemData::Client const &client);

    // Construct from attributes of the given vehicle type.
    DurationSegment(size_t depot, ProblemData::VehicleType const &vehicleType);

    // Construct from raw data.
    inline DurationSegment(size_t idxFirst,
                           size_t idxLast,
                           Duration duration,
                           Duration timeWarp,
                           Duration twEarly,
                           Duration twLate,
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
    // Because clients' default time windows are [0, INT_MAX], the ternaries in
    // this method are carefully designed to avoid integer over- and underflow
    // issues. Be very careful when changing things here!
    using Dur = pyvrp::Duration;

    // edgeDuration is the travel duration from our last to the other's first
    // client, and atOther the time (relative to our starting time) at which we
    // arrive there.
    Dur const edgeDuration = durationMatrix(idxLast_, other.idxFirst_);
    Dur const atOther = duration_ - timeWarp_ + edgeDuration;

    // Time warp increases when we arrive at other after its time window closes.
    Dur const diffTw = twEarly_ + atOther > other.twLate_
                           ? twEarly_ + atOther - other.twLate_
                           : 0;

    // Wait duration increases if we arrive there before its time window opens.
    Dur const diffWait = other.twEarly_ - atOther > twLate_
                             ? other.twEarly_ - atOther - twLate_
                             : 0;

    Dur const otherLate  // new twLate for the other segment
        = atOther > other.twLate_ - std::numeric_limits<Dur>::max()
              ? other.twLate_ - atOther
              : other.twLate_;

    return {idxFirst_,
            other.idxLast_,
            duration_ + other.duration_ + edgeDuration + diffWait,
            timeWarp_ + other.timeWarp_ + diffTw,
            std::max(other.twEarly_ - atOther, twEarly_) - diffWait,
            std::min(otherLate, twLate_) + diffTw,
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
    return timeWarp_
           + std::max<Duration>(releaseTime_ - twLate_, 0)
           // Max duration constraint applies only to net route duration,
           // subtracting existing time warp. Use ternary to avoid underflow.
           + (duration_ - timeWarp_ > maxDuration
                  ? duration_ - timeWarp_ - maxDuration
                  : 0);
}

DurationSegment::DurationSegment(size_t idxFirst,
                                 size_t idxLast,
                                 Duration duration,
                                 Duration timeWarp,
                                 Duration twEarly,
                                 Duration twLate,
                                 Duration releaseTime)
    : idxFirst_(idxFirst),
      idxLast_(idxLast),
      duration_(duration),
      timeWarp_(timeWarp),
      twEarly_(twEarly),
      twLate_(twLate),
      releaseTime_(releaseTime)
{
}
}  // namespace pyvrp

#endif  // PYVRP_DURATIONSEGMENT_H
