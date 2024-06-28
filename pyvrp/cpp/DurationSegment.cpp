#include "DurationSegment.h"

#include <cassert>

using pyvrp::Duration;
using pyvrp::DurationSegment;

Duration DurationSegment::twEarly() const { return twEarly_; }

Duration DurationSegment::twLate() const { return twLate_; }

Duration DurationSegment::releaseTime() const { return releaseTime_; }

DurationSegment::DurationSegment(size_t idx, ProblemData::Client const &client)
    : idxFirst_(idx),
      idxLast_(idx),
      duration_(client.serviceDuration),
      timeWarp_(0),
      twEarly_(client.twEarly),
      twLate_(client.twLate),
      releaseTime_(client.releaseTime)
{
}

DurationSegment::DurationSegment(size_t depot,
                                 ProblemData::VehicleType const &vehicleType)
    : idxFirst_(depot),
      idxLast_(depot),
      duration_(0),
      timeWarp_(0),
      twEarly_(vehicleType.twEarly),
      twLate_(vehicleType.twLate),
      releaseTime_(0)
{
    assert(depot == vehicleType.startDepot || depot == vehicleType.endDepot);
}
