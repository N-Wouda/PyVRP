#ifndef PYVRP_SOLUTION_H
#define PYVRP_SOLUTION_H

#include "Measure.h"
#include "ProblemData.h"
#include "RandomNumberGenerator.h"

#include <functional>
#include <iosfwd>
#include <optional>
#include <vector>

namespace pyvrp
{
/**
 * Solution(data: ProblemData, routes: Union[list[Route], list[list[int]]])
 *
 * Encodes VRP solutions.
 *
 * Parameters
 * ----------
 * data
 *     Data instance.
 * routes
 *     Route list to use. Can be a list of :class:`~Route` objects, or a lists
 *     of client visits. In case of the latter, all routes are assigned
 *     vehicles of the first type. That need not be a feasible assignment!
 *
 * Raises
 * ------
 * RuntimeError
 *     When the given solution is invalid in one of several ways. In
 *     particular when the number of routes in the ``routes`` argument exceeds
 *     :py:attr:`~ProblemData.num_vehicles`, when an empty route has been
 *     passed as part of ``routes``, when too many vehicles of a particular
 *     type have been used, or when a client is visited more than once.
 */
class Solution
{
    using Client = size_t;
    using Depot = size_t;
    using VehicleType = size_t;

public:
    /**
     * Route(data: ProblemData, visits: list[int], vehicle_type: int)
     *
     * A simple class that stores the route plan and some statistics.
     */
    class Route
    {
        using Visits = std::vector<Client>;

        Visits visits_ = {};           // Client visits on this route
        Distance distance_ = 0;        // Total travel distance on this route
        Cost distanceCost_ = 0;        // Total cost of travel distance
        Distance excessDistance_ = 0;  // Excess travel distance
        Load delivery_ = 0;       // Total delivery amount served on this route
        Load pickup_ = 0;         // Total pickup amount gathered on this route
        Load excessLoad_ = 0;     // Excess pickup or delivery demand
        Duration duration_ = 0;   // Total duration of this route
        Cost durationCost_ = 0;   // Total cost of route duration
        Duration timeWarp_ = 0;   // Total time warp on this route
        Duration travel_ = 0;     // Total *travel* duration on this route
        Duration service_ = 0;    // Total *service* duration on this route
        Duration wait_ = 0;       // Total *waiting* duration on this route
        Duration release_ = 0;    // Release time of this route
        Duration startTime_ = 0;  // (earliest) start time of this route
        Duration slack_ = 0;      // Total time slack on this route
        Cost prizes_ = 0;         // Total value of prizes on this route

        std::pair<double, double> centroid_;  // Route center
        VehicleType vehicleType_;             // Type of vehicle
        Depot startDepot_;                    // Assigned start depot
        Depot endDepot_;                      // Assigned end depot

    public:
        [[nodiscard]] bool empty() const;

        /**
         * Returns the number of clients visited by this route.
         */
        [[nodiscard]] size_t size() const;

        [[nodiscard]] Client operator[](size_t idx) const;

        Visits::const_iterator begin() const;
        Visits::const_iterator end() const;

        /**
         * Route visits, as a list of clients.
         */
        [[nodiscard]] Visits const &visits() const;

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
         * the route can leave the depot and have a minimal duration and time
         * warp. If there is positive :meth:`~slack`, the start time can be
         * delayed by at most :meth:`~slack` time units without increasing the
         * total (minimal) route duration, or time warp.
         *
         * .. note::
         *
         *    It may be possible to leave before the start time (if the depot
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
         * Time by which departure from the depot can be delayed without
         * resulting in (additional) time warp or increased route duration.
         */
        [[nodiscard]] Duration slack() const;

        /**
         * Earliest time at which this route can leave the depot. Follows from
         * the release times of clients visited on this route.
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

        Route(ProblemData const &data,
              Visits visits,
              VehicleType const vehicleType);

        // This constructor does *no* validation. Useful when unserialising
        // objects.
        Route(Visits visits,
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
              Duration wait,
              Duration release,
              Duration startTime,
              Duration slack,
              Cost prizes,
              std::pair<double, double> centroid,
              VehicleType vehicleType,
              Depot startDepot,
              Depot endDepot);
    };

private:
    using Routes = std::vector<Route>;
    using Neighbours = std::vector<std::optional<std::pair<Client, Client>>>;

    size_t numClients_ = 0;         // Number of clients in the solution
    size_t numMissingClients_ = 0;  // Number of required but missing clients
    Distance distance_ = 0;         // Total travel distance over all routes
    Cost distanceCost_ = 0;         // Total cost of all routes' travel distance
    Duration duration_ = 0;         // Total duration over all routes
    Cost durationCost_ = 0;         // Total cost of all routes' duration
    Distance excessDistance_ = 0;   // Total excess distance over all routes
    Load excessLoad_ = 0;           // Total excess load over all routes
    Cost fixedVehicleCost_ = 0;     // Fixed cost of all used vehicles
    Cost prizes_ = 0;               // Total collected prize value
    Cost uncollectedPrizes_ = 0;    // Total uncollected prize value
    Duration timeWarp_ = 0;         // Total time warp over all routes
    bool isGroupFeas_ = true;       // Is feasible w.r.t. client groups?

    Routes routes_;
    Neighbours neighbours_;  // client [pred, succ] pairs, null if unassigned

    // Determines the [pred, succ] pairs for assigned clients.
    void makeNeighbours(ProblemData const &data);

    // Evaluates this solution's characteristics.
    void evaluate(ProblemData const &data);

    // These are only available within a solution; from the outside a solution
    // is immutable.
    Solution &operator=(Solution const &other) = default;
    Solution &operator=(Solution &&other) = default;

public:
    // Solution is empty when it has no routes and no clients.
    [[nodiscard]] bool empty() const;

    /**
     * Number of routes in this solution.
     */
    [[nodiscard]] size_t numRoutes() const;

    /**
     * Number of clients in this solution.
     *
     * .. warning::
     *
     *    An empty solution typically indicates that there is a significant
     *    difference between the values of the prizes of the optional clients
     *    and the other objective terms. This hints at a scaling issue in the
     *    data.
     */
    [[nodiscard]] size_t numClients() const;

    /**
     * Number of required clients that are not in this solution.
     */
    [[nodiscard]] size_t numMissingClients() const;

    /**
     * The solution's routing decisions.
     *
     * Returns
     * -------
     * list
     *     A list of routes. Each :class:`~Route` starts and ends at a depot,
     *     but that is implicit: the depot is not part of the returned routes.
     */
    [[nodiscard]] Routes const &routes() const;

    /**
     * Returns a list of neighbours for each client, by index.
     *
     * Returns
     * -------
     * list
     *     A list of ``(pred, succ)`` tuples that encode for each client their
     *     predecessor and successors in this solutions's routes. ``None`` in
     *     case the client is not in the solution (or is a depot).
     */
    [[nodiscard]] Neighbours const &neighbours() const;

    /**
     * Whether this solution is feasible.
     */
    [[nodiscard]] bool isFeasible() const;

    /**
     * Returns whether this solution is feasible w.r.t. the client group
     * restrictions.
     */
    [[nodiscard]] bool isGroupFeasible() const;

    /**
     * Returns whether this solution is complete, which it is when it has all
     * required clients.
     */
    [[nodiscard]] bool isComplete() const;

    /**
     * Returns whether this solution violates capacity constraints.
     */
    [[nodiscard]] bool hasExcessLoad() const;

    /**
     * Returns whether this solution violates maximum distance constraints.
     *
     * Returns
     * -------
     * bool
     *     True if the solution is not feasible with respect to the maximum
     *     distance constraints of the vehicles servicing routes in this
     *     solution. False otherwise.
     */
    [[nodiscard]] bool hasExcessDistance() const;

    /**
     * Returns whether this solution violates time window or maximum duration
     * constraints.
     */
    [[nodiscard]] bool hasTimeWarp() const;

    /**
     * Returns the total distance over all routes.
     */
    [[nodiscard]] Distance distance() const;

    /**
     * Total cost of the distance travelled on routes in this solution.
     */
    [[nodiscard]] Cost distanceCost() const;

    /**
     * Total duration of all routes in this solution.
     */
    [[nodiscard]] Duration duration() const;

    /**
     * Total cost of the duration of all routes in this solution.
     */
    [[nodiscard]] Cost durationCost() const;

    /**
     * Returns the total excess load over all routes.
     */
    [[nodiscard]] Load excessLoad() const;

    /**
     * Returns the total distance in excess of maximum duration constraints,
     * over all routes.
     */
    [[nodiscard]] Distance excessDistance() const;

    /**
     * Returns the fixed vehicle cost of all vehicles used in this solution.
     */
    [[nodiscard]] Cost fixedVehicleCost() const;

    /**
     * Returns the total collected prize value over all routes.
     */
    [[nodiscard]] Cost prizes() const;

    /**
     * Total prize value of all clients not visited in this solution.
     */
    [[nodiscard]] Cost uncollectedPrizes() const;

    /**
     * Returns the total time warp load over all routes.
     */
    [[nodiscard]] Duration timeWarp() const;

    bool operator==(Solution const &other) const;

    Solution(Solution const &other) = default;
    Solution(Solution &&other) = default;

    /**
     * make_random(data: ProblemData, rng: RandomNumberGenerator) -> Solution
     *
     * Creates a randomly generated solution.
     *
     * Parameters
     * ----------
     * data
     *     Data instance.
     * rng
     *     Random number generator to use.
     *
     * Returns
     * -------
     * Solution
     *     The randomly generated solution.
     */
    Solution(ProblemData const &data, RandomNumberGenerator &rng);

    /**
     * Constructs a solution using routes given as lists of client indices.
     * This constructor assumes all routes use vehicles having vehicle type 0.
     *
     * @param data   Data instance describing the problem that's being solved.
     * @param routes Solution's route list.
     */
    Solution(ProblemData const &data,
             std::vector<std::vector<Client>> const &routes);

    /**
     * Constructs a solution from the given list of Routes.
     *
     * @param data   Data instance describing the problem that's being solved.
     * @param routes Solution's route list.
     */
    Solution(ProblemData const &data, Routes const &routes);

    // This constructor does *no* validation. Useful when unserialising objects.
    Solution(size_t numClients,
             size_t numMissingClients,
             Distance distance,
             Cost distanceCost,
             Duration duration,
             Cost durationCost,
             Distance excessDistance,
             Load excessLoad,
             Cost fixedVehicleCost,
             Cost prizes,
             Cost uncollectedPrizes,
             Duration timeWarp,
             bool isGroupFeasible,
             Routes const &routes,
             Neighbours neighbours);
};
}  // namespace pyvrp

std::ostream &operator<<(std::ostream &out, pyvrp::Solution const &sol);
std::ostream &operator<<(std::ostream &out,
                         pyvrp::Solution::Route const &route);

template <> struct std::hash<pyvrp::Solution>
{
    size_t operator()(pyvrp::Solution const &sol) const
    {
        size_t res = 17;
        res = res * 31 + std::hash<size_t>()(sol.numRoutes());
        res = res * 31 + std::hash<pyvrp::Distance>()(sol.distance());
        res = res * 31 + std::hash<pyvrp::Load>()(sol.excessLoad());
        res = res * 31 + std::hash<pyvrp::Duration>()(sol.timeWarp());

        return res;
    }
};

#endif  // PYVRP_SOLUTION_H
