#include "DurationSegment.h"

using pyvrp::Duration;
using pyvrp::DurationSegment;

Duration DurationSegment::earliestStart() const { return earliestStart_; }

Duration DurationSegment::releaseTime() const { return releaseTime_; }

DurationSegment::DurationSegment(size_t idx, ProblemData::Client const &client)
    : idxFirst_(idx),
      idxLast_(idx),
      duration_(client.serviceDuration),
      earliestStart_(client.twEarly),
      latestFinish_(client.twLate > std::numeric_limits<Duration>::max()
                                        - client.serviceDuration
                        ? std::numeric_limits<Duration>::max()
                        : client.twLate + client.serviceDuration),
      releaseTime_(client.releaseTime)
{
}
