#ifndef PYVRP_ROUTE_H
#define PYVRP_ROUTE_H

#include "Measure.h"
#include "ProblemData.h"
#include "RandomNumberGenerator.h"

#include <iosfwd>
#include <optional>
#include <vector>

namespace pyvrp
{
/**
 * Route(
 *     data: ProblemData,
 *     visits: Union[list[int], list[list[int]]],
 *     vehicle_type: int,
 * )
 *
 * A simple class that stores the client ``visits`` and some statistics. The
 * client visits can be organised into multiple distinct trips, separated by a
 * depot visit.
 */
class Route
{
    using Client = size_t;
    using Depot = size_t;
    using VehicleType = size_t;

    using Trip = std::vector<Client>;
    using Trips = std::vector<Trip>;

    /**
     * Forward iterator through the clients visited by this route.
     */
    class Iterator
    {
        Trips const *trips = nullptr;
        size_t trip = 0;   // trip index in trips
        size_t visit = 0;  // visit index into trips[trip]

        Iterator(Trips const &trips, size_t trip, size_t visit);

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Client;

        Iterator() = default;
        Iterator(Iterator const &other) = default;
        Iterator(Iterator &&other) = default;

        Iterator &operator=(Iterator const &other) = default;
        Iterator &operator=(Iterator &&other) = default;

        static Iterator begin(Trips const &trips);
        static Iterator end(Trips const &trips);

        bool operator==(Iterator const &other) const;

        Client operator*() const;

        Iterator operator++(int);
        Iterator &operator++();
    };

    Trips trips_;                  // Trips that make up this route
    Distance distance_ = 0;        // Total travel distance on this route
    Cost distanceCost_ = 0;        // Total cost of travel distance
    Distance excessDistance_ = 0;  // Excess travel distance
    Load delivery_ = 0;            // Total delivery amount served on this route
    Load pickup_ = 0;              // Total pickup amount gathered on this route
    Load excessLoad_ = 0;          // Excess pickup or delivery demand
    Duration duration_ = 0;        // Total duration of this route
    Cost durationCost_ = 0;        // Total cost of route duration
    Duration timeWarp_ = 0;        // Total time warp on this route
    Duration travel_ = 0;          // Total *travel* duration on this route
    Duration service_ = 0;         // Total *service* duration on this route
    Duration release_ = 0;         // Release time of this route
    Duration startTime_ = 0;       // (earliest) start time of this route
    Duration slack_ = 0;           // Total time slack on this route
    Cost prizes_ = 0;              // Total value of prizes on this route

    std::pair<double, double> centroid_;  // Route center
    VehicleType vehicleType_;             // Type of vehicle
    Depot startDepot_;                    // Assigned start depot
    Depot endDepot_;                      // Assigned end depot
    std::optional<Depot> reloadDepot_;    // Optionally assigned reload depot

public:
    [[nodiscard]] bool empty() const;

    /**
     * Returns the number of clients visited by this route.
     */
    [[nodiscard]] size_t size() const;

    [[nodiscard]] Client operator[](size_t idx) const;

    Iterator begin() const;
    Iterator end() const;

    /**
     * Route visits, as a list of clients.
     */
    [[nodiscard]] std::vector<Client> visits() const;

    /**
     * List of trips that constitute this route.
     */
    [[nodiscard]] Trips const &trips() const;

    /**
     * Returns the trip data of the given trip index.
     */
    [[nodiscard]] Trip const &trip(size_t trip) const;

    /**
     * Number of trips in this route.
     */
    [[nodiscard]] size_t numTrips() const;

    /**
     * Total distance travelled on this route.
     */
    [[nodiscard]] Distance distance() const;

    /**
     * Total cost of the distance travelled on this route.
     */
    [[nodiscard]] Cost distanceCost() const;

    /**
     * Distance in excess of the vehicle's maximum distance constraint.
     */
    [[nodiscard]] Distance excessDistance() const;

    /**
     * Total client delivery load on this route.
     */
    [[nodiscard]] Load delivery() const;

    /**
     * Total client pickup load on this route.
     */
    [[nodiscard]] Load pickup() const;

    /**
     * Pickup or delivery load in excess of the vehicle's capacity.
     */
    [[nodiscard]] Load excessLoad() const;

    /**
     * Total route duration, including travel, service and waiting time.
     */
    [[nodiscard]] Duration duration() const;

    /**
     * Total cost of the duration of this route.
     */
    [[nodiscard]] Cost durationCost() const;

    /**
     * Total duration of service on this route.
     */
    [[nodiscard]] Duration serviceDuration() const;

    /**
     * Amount of time warp incurred on this route.
     */
    [[nodiscard]] Duration timeWarp() const;

    /**
     * Total duration of travel on this route.
     */
    [[nodiscard]] Duration travelDuration() const;

    /**
     * Total waiting duration on this route.
     */
    [[nodiscard]] Duration waitDuration() const;

    /**
     * Start time of this route. This is the earliest possible time at which
     * the route can leave the depot and have a minimal duration and time warp.
     * If there is positive :meth:`~slack`, the start time can be delayed by at
     * most :meth:`~slack` time units without increasing the total (minimal)
     * route duration, or time warp.
     *
     * .. note::
     *
     *    It may be possible to leave before the start time (if the vehicle's
     *    time window allows for it). That will introduce additional waiting
     *    time, such that the route duration will then no longer be minimal.
     *    Delaying departure by more than :meth:`~slack` time units always
     *    increases time warp, which could turn the route infeasible.
     */
    [[nodiscard]] Duration startTime() const;

    /**
     * End time of the route. This is equivalent to
     * ``start_time + duration - time_warp``.
     */
    [[nodiscard]] Duration endTime() const;

    /**
     * Time by which departure from the depot can be delayed without resulting
     * in (additional) time warp or increased route duration.
     */
    [[nodiscard]] Duration slack() const;

    /**
     * Earliest time at which this route can leave the depot. Follows from the
     * release times of clients visited on this route.
     *
     * .. note::
     *
     *    The route's release time should not be later than its start time,
     *    unless the route has time warp.
     */
    [[nodiscard]] Duration releaseTime() const;

    /**
     * Total prize value collected on this route.
     */
    [[nodiscard]] Cost prizes() const;

    /**
     * Center point of the client locations on this route.
     */
    [[nodiscard]] std::pair<double, double> const &centroid() const;

    /**
     * Index of the type of vehicle used on this route.
     */
    [[nodiscard]] VehicleType vehicleType() const;

    /**
     * Location index of the route's starting depot.
     */
    [[nodiscard]] Depot startDepot() const;

    /**
     * Location index of the route's ending depot.
     */
    [[nodiscard]] Depot endDepot() const;

    /**
     * Optional location index of the route's reload depot, if available.
     */
    [[nodiscard]] std::optional<Depot> reloadDepot() const;

    /**
     * Returns whether this route is feasible.
     */
    [[nodiscard]] bool isFeasible() const;

    /**
     * Returns whether this route violates capacity constraints.
     */
    [[nodiscard]] bool hasExcessLoad() const;

    /**
     * Returns whether this route violates maximum distance constraints.
     */
    [[nodiscard]] bool hasExcessDistance() const;

    /**
     * Returns whether this route violates time window or maximum duration
     * constraints.
     */
    [[nodiscard]] bool hasTimeWarp() const;

    bool operator==(Route const &other) const;

    Route() = delete;

    // Case where the visits are made in a single trip.
    Route(ProblemData const &data, Trip visits, VehicleType vehicleType);

    // Case where the visits are made over one or more trips.
    Route(ProblemData const &data, Trips visits, VehicleType vehicleType);

    // This constructor does *no* validation. Useful when unserialising objects.
    Route(Trips trips,
          Distance distance,
          Cost distanceCost,
          Distance excessDistance,
          Load delivery,
          Load pickup,
          Load excessLoad,
          Duration duration,
          Cost durationCost,
          Duration timeWarp,
          Duration travel,
          Duration service,
          Duration release,
          Duration startTime,
          Duration slack,
          Cost prizes,
          std::pair<double, double> centroid,
          VehicleType vehicleType,
          Depot startDepot,
          Depot endDepot,
          std::optional<Depot> reloadDepot);
};
}  // namespace pyvrp

std::ostream &operator<<(std::ostream &out, pyvrp::Route const &route);

#endif  // PYVRP_ROUTE_H
