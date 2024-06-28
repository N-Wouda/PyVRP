#ifndef PYVRP_ROUTE_H
#define PYVRP_ROUTE_H

#include "DistanceSegment.h"
#include "DurationSegment.h"
#include "LoadSegment.h"
#include "ProblemData.h"

#include <cassert>
#include <iosfwd>

namespace pyvrp::search
{
/**
 * This ``Route`` class supports fast delta cost computations and in-place
 * modification. It can be used to implement move evaluations.
 *
 * A ``Route`` object tracks a full route, including the depots. The clients
 * and depots on the route can be accessed using ``Route::operator[]`` on a
 * ``route`` object: ``route[0]`` and ``route[route.size() + 1]`` are the start
 * and end depots, respectively, and any clients in between are on the indices
 * ``{1, ..., size()}`` (empty if ``size() == 0``). Note that ``Route::size()``
 * returns the number of *clients* in the route; this excludes the depots.
 *
 * .. note::
 *
 *    Modifications to the ``Route`` object do not immediately propagate to its
 *    statistics like time window, load and distance data. To make that happen,
 *    ``Route::update()`` must be called!
 */
class Route
{
    // These classes (defined further below) handle transparent access to the
    // route's segment-specific concatenation schemes.
    friend class SegmentAt;
    friend class SegmentAfter;
    friend class SegmentBefore;
    friend class SegmentBetween;

public:
    /**
     * A simple class that tracks a new proposed structure for a given route.
     * This new structure can be efficiently evaluated by calling appropriate
     * member functions to return concatenation schemes that detail the
     * newly proposed route's statistics.
     */
    template <typename... Segments> class Proposal
    {
        Route const *current;
        ProblemData const &data;
        std::tuple<Segments...> segments;

    public:
        Proposal(Route const *current,
                 ProblemData const &data,
                 Segments &&...segments);

        Route const *route() const;

        DistanceSegment distanceSegment() const;
        DurationSegment durationSegment() const;
        LoadSegment loadSegment() const;
    };

    /**
     * Light wrapper class around a client or depot location. This class tracks
     * the route it is in, and the position and role it currently has in that
     * route.
     */
    class Node
    {
        friend class Route;

        size_t loc_;    // Location represented by this node
        size_t idx_;    // Position in the route
        Route *route_;  // Indicates membership of a route, if any

    public:
        Node(size_t loc);

        /**
         * Returns the location represented by this node.
         */
        [[nodiscard]] inline size_t client() const;  // TODO rename to loc

        /**
         * Returns this node's position in a route. This value is ``0`` when
         * the node is *not* in a route.
         */
        [[nodiscard]] inline size_t idx() const;

        /**
         * Returns the route this node is currently in. If the node is not in
         * a route, this returns ``None`` (C++: ``nullptr``).
         */
        [[nodiscard]] inline Route *route() const;

        /**
         * Returns whether this node is a depot. A node can only be a depot if
         * it is in a route.
         */
        [[nodiscard]] inline bool isDepot() const;
    };

private:
    /**
     * Class storing data related to the route location at ``idx``.
     */
    class SegmentAt
    {
        Route const *route;
        size_t const idx;

    public:
        inline SegmentAt(Route const &route, size_t idx);
        inline DistanceSegment distance(size_t profile) const;
        inline DurationSegment duration(size_t profile) const;
        inline LoadSegment load() const;
    };

    /**
     * Class storing data related to the route segment starting at ``start``,
     * and ending at the depot (inclusive).
     */
    class SegmentAfter
    {
        Route const *route;
        size_t const start;

    public:
        inline SegmentAfter(Route const &route, size_t start);
        inline DistanceSegment distance(size_t profile) const;
        inline DurationSegment duration(size_t profile) const;
        inline LoadSegment load() const;
    };

    /**
     * Class storing data related to the route segment starting at the depot,
     * and ending at ``end`` (inclusive).
     */
    class SegmentBefore
    {
        Route const *route;
        size_t const end;

    public:
        inline SegmentBefore(Route const &route, size_t end);
        inline DistanceSegment distance(size_t profile) const;
        inline DurationSegment duration(size_t profile) const;
        inline LoadSegment load() const;
    };

    /**
     * Class storing data related to the route segment starting at ``start``,
     * and ending at ``end`` (inclusive).
     */
    class SegmentBetween
    {
        Route const *route;
        size_t const start;
        size_t const end;

    public:
        inline SegmentBetween(Route const &route, size_t start, size_t end);
        inline DistanceSegment distance(size_t profile) const;
        inline DurationSegment duration(size_t profile) const;
        inline LoadSegment load() const;
    };

    ProblemData const &data;

    // Cache the vehicle type object here. Since the vehicle type's properties
    // are called quite often, it's much better to have this object readily
    // available, rather than take it by reference.
    ProblemData::VehicleType const vehicleType_;
    size_t const vehTypeIdx_;
    size_t const idx_;

    std::vector<Node *> nodes;  // Nodes in this route, including depots
    std::pair<double, double> centroid_;  // Center point of route's clients

    Node startDepot_;  // Departure depot for this route
    Node endDepot_;    // Return depot for this route

    std::vector<DistanceSegment> distAt;      // Dist data at each node
    std::vector<DistanceSegment> distBefore;  // Dist of depot -> client (incl.)
    std::vector<DistanceSegment> distAfter;   // Dist of client -> depot (incl.)

    std::vector<LoadSegment> loadAt;      // Load data at each node
    std::vector<LoadSegment> loadAfter;   // Load of client -> depot (incl)
    std::vector<LoadSegment> loadBefore;  // Load of depot -> client (incl)

    std::vector<DurationSegment> durAt;      // Duration data at each node
    std::vector<DurationSegment> durAfter;   // Dur of client -> depot (incl.)
    std::vector<DurationSegment> durBefore;  // Dur of depot -> client (incl.)

#ifndef NDEBUG
    // When debug assertions are enabled, we use this flag to check whether
    // the statistics are still in sync with the route's nodes list. Statistics
    // are only updated after calling ``update()``. If that function has not
    // yet been called after inserting or removing nodes, this flag is active,
    // and asserts on statistics getters will fail.
    bool dirty = false;
#endif

public:
    /**
     * Route index.
     */
    [[nodiscard]] inline size_t idx() const;

    /**
     * @return The client or depot node at the given ``idx``.
     */
    [[nodiscard]] inline Node *operator[](size_t idx);

    // First client in the route if the route is non-empty. Else it is the
    // end depot. In either case the iterator is valid!
    [[nodiscard]] std::vector<Node *>::const_iterator begin() const;
    [[nodiscard]] std::vector<Node *>::iterator begin();

    // End depot. The iterator is valid!
    [[nodiscard]] std::vector<Node *>::const_iterator end() const;
    [[nodiscard]] std::vector<Node *>::iterator end();

    /**
     * Returns a route proposal object that stores the given route segment
     * arguments.
     */
    template <typename... Segments>
    [[nodiscard]] Proposal<Segments...> proposal(Segments &&...segments) const;

    /**
     * Tests if this route is feasible.
     *
     * @return true if the route is feasible, false otherwise.
     */
    [[nodiscard]] inline bool isFeasible() const;

    /**
     * Determines whether this route is load-feasible.
     *
     * @return true if the route exceeds the capacity, false otherwise.
     */
    [[nodiscard]] inline bool hasExcessLoad() const;

    /**
     * Determines whether this route is distance-feasible.
     *
     * @return true if the route exceeds the maximum distance constraint, false
     *         otherwise.
     */
    [[nodiscard]] inline bool hasExcessDistance() const;

    /**
     * Determines whether this route is time-feasible.
     *
     * @return true if the route has time warp, false otherwise.
     */
    [[nodiscard]] inline bool hasTimeWarp() const;

    /**
     * @return Total load on this route.
     */
    [[nodiscard]] inline Load load() const;

    /**
     * Load (as a consequence of pickup and deliveries) in excess of the
     * vehicle's capacity.
     */
    [[nodiscard]] inline Load excessLoad() const;

    /**
     * Travel distance in excess of the assigned vehicle type's maximum
     * distance constraint.
     */
    [[nodiscard]] inline Distance excessDistance() const;

    /**
     * @return The load capacity of the vehicle servicing this route.
     */
    [[nodiscard]] inline Load capacity() const;

    /**
     * @return The location index of this route's starting depot.
     */
    [[nodiscard]] inline size_t startDepot() const;

    /**
     * @return The location index of this route's ending depot.
     */
    [[nodiscard]] inline size_t endDepot() const;

    /**
     * @return The fixed cost of the vehicle servicing this route.
     */
    [[nodiscard]] inline Cost fixedVehicleCost() const;

    /**
     * @return Total distance travelled on this route.
     */
    [[nodiscard]] inline Distance distance() const;

    /**
     * @return Cost of the distance travelled on this route.
     */
    [[nodiscard]] inline Cost distanceCost() const;

    /**
     * @return Cost per unit of distance travelled on this route.
     */
    [[nodiscard]] inline Cost unitDistanceCost() const;

    /**
     * @return The duration of this route.
     */
    [[nodiscard]] inline Duration duration() const;

    /**
     * @return Cost of this route's duration.
     */
    [[nodiscard]] inline Cost durationCost() const;

    /**
     * @return Cost per unit of duration travelled on this route.
     */
    [[nodiscard]] inline Cost unitDurationCost() const;

    /**
     * @return The maximum route duration that the vehicle servicing this route
     *         supports.
     */
    [[nodiscard]] inline Duration maxDuration() const;

    /**
     * @return The maximum route distance that the vehicle servicing this route
     *         supports.
     */
    [[nodiscard]] inline Distance maxDistance() const;

    /**
     * @return Total time warp on this route.
     */
    [[nodiscard]] inline Duration timeWarp() const;

    /**
     * @return The routing profile of the vehicle servicing this route.
     */
    [[nodiscard]] inline size_t profile() const;

    /**
     * @return true if this route is empty, false otherwise.
     */
    [[nodiscard]] inline bool empty() const;

    /**
     * @return Number of clients in this route.
     */
    [[nodiscard]] inline size_t size() const;

    /**
     * Returns an object that can be queried for data associated with the node
     * at idx.
     */
    [[nodiscard]] inline SegmentAt at(size_t idx) const;

    /**
     * Returns an object that can be queried for data associated with the
     * segment starting at start.
     */
    [[nodiscard]] inline SegmentAfter after(size_t start) const;

    /**
     * Returns an object that can be queried for data associated with the
     * segment ending at end.
     */
    [[nodiscard]] inline SegmentBefore before(size_t end) const;

    /**
     * Returns an object that can be queried for data associated with the
     * segment between [start, end].
     */
    [[nodiscard]] inline SegmentBetween between(size_t start, size_t end) const;

    /**
     * Center point of the client locations on this route.
     */
    [[nodiscard]] std::pair<double, double> const &centroid() const;

    /**
     * @return This route's vehicle type.
     */
    [[nodiscard]] size_t vehicleType() const;

    /**
     * Tests if this route potentially overlaps with the other route, subject
     * to a tolerance in [0, 1].
     */
    [[nodiscard]] bool overlapsWith(Route const &other, double tolerance) const;

    /**
     * Clears all clients on this route. After calling this method, ``empty()``
     * returns true and ``size()`` is zero.
     */
    void clear();

    /**
     * Inserts the given node at index ``idx``. Assumes the given index is
     * valid.
     */
    void insert(size_t idx, Node *node);

    /**
     * Inserts the given node at the back of the route.
     */
    void push_back(Node *node);

    /**
     * Removes the node at ``idx`` from the route.
     */
    void remove(size_t idx);

    /**
     * Swaps the given nodes.
     */
    static void swap(Node *first, Node *second);

    /**
     * Updates this route. To be called after swapping nodes/changing the
     * solution.
     */
    void update();

    Route(ProblemData const &data, size_t idx, size_t vehicleType);
    ~Route();
};

/**
 * Convenience method accessing the node directly before the argument.
 */
inline Route::Node *p(Route::Node *node)
{
    auto &route = *node->route();
    return route[node->idx() - 1];
}

/**
 * Convenience method accessing the node directly after the argument.
 */
inline Route::Node *n(Route::Node *node)
{
    auto &route = *node->route();
    return route[node->idx() + 1];
}

size_t Route::Node::client() const { return loc_; }

size_t Route::Node::idx() const { return idx_; }

Route *Route::Node::route() const { return route_; }

bool Route::Node::isDepot() const
{
    // We need to be in a route to be the depot. If we are, then we need to
    // be either the route's start or end depot.
    return route_ && (idx_ == 0 || idx_ == route_->size() + 1);
}

Route::SegmentAt::SegmentAt(Route const &route, size_t idx)
    : route(&route), idx(idx)
{
    assert(idx < route.nodes.size());
}

Route::SegmentAfter::SegmentAfter(Route const &route, size_t start)
    : route(&route), start(start)
{
    assert(start < route.nodes.size());
}

Route::SegmentBefore::SegmentBefore(Route const &route, size_t end)
    : route(&route), end(end)
{
    assert(end < route.nodes.size());
}

Route::SegmentBetween::SegmentBetween(Route const &route,
                                      size_t start,
                                      size_t end)
    : route(&route), start(start), end(end)
{
    assert(start <= end && end < route.nodes.size());
}

DistanceSegment
Route::SegmentAt::distance([[maybe_unused]] size_t profile) const
{
    return route->distAt[idx];
}

DurationSegment
Route::SegmentAt::duration([[maybe_unused]] size_t profile) const
{
    return route->durAt[idx];
}

LoadSegment Route::SegmentAt::load() const { return route->loadAt[idx]; }

DistanceSegment Route::SegmentAfter::distance(size_t profile) const
{
    if (profile == route->profile())
        return route->distAfter[start];

    auto const between = SegmentBetween(*route, start, route->size() + 1);
    return between.distance(profile);
}

DurationSegment Route::SegmentAfter::duration(size_t profile) const
{
    if (profile == route->profile())
        return route->durAfter[start];

    auto const between = SegmentBetween(*route, start, route->size() + 1);
    return between.duration(profile);
}

LoadSegment Route::SegmentAfter::load() const
{
    return route->loadAfter[start];
}

DistanceSegment Route::SegmentBefore::distance(size_t profile) const
{
    if (profile == route->profile())
        return route->distBefore[end];

    auto const between = SegmentBetween(*route, size_t(0), end);
    return between.distance(profile);
}

DurationSegment Route::SegmentBefore::duration(size_t profile) const
{
    if (profile == route->profile())
        return route->durBefore[end];

    auto const between = SegmentBetween(*route, size_t(0), end);
    return between.duration(profile);
}

LoadSegment Route::SegmentBefore::load() const
{
    return route->loadBefore[end];
}

DistanceSegment Route::SegmentBetween::distance(size_t profile) const
{
    if (profile != route->profile())  // then we have to compute the distance
    {                                 // segment from scratch.
        auto distSegment = route->distAt[start];

        for (size_t step = start; step != end; ++step)
        {
            auto const &mat = route->data.distanceMatrix(profile);
            auto const &distAt = route->distAt[step + 1];
            distSegment = DistanceSegment::merge(mat, distSegment, distAt);
        }

        return distSegment;
    }

    auto const &startDist = route->distBefore[start];
    auto const &endDist = route->distBefore[end];

    assert(startDist.distance() <= endDist.distance());
    return DistanceSegment(route->nodes[start]->client(),
                           route->nodes[end]->client(),
                           endDist.distance() - startDist.distance());
}

DurationSegment Route::SegmentBetween::duration(size_t profile) const
{
    auto durSegment = route->durAt[start];

    for (size_t step = start; step != end; ++step)
    {
        auto const &mat = route->data.durationMatrix(profile);
        auto const &durAt = route->durAt[step + 1];
        durSegment = DurationSegment::merge(mat, durSegment, durAt);
    }

    return durSegment;
}

LoadSegment Route::SegmentBetween::load() const
{
    auto loadSegment = route->loadAt[start];

    for (size_t step = start; step != end; ++step)
        loadSegment = LoadSegment::merge(loadSegment, route->loadAt[step + 1]);

    return loadSegment;
}

bool Route::isFeasible() const
{
    assert(!dirty);
    return !hasExcessLoad() && !hasTimeWarp() && !hasExcessDistance();
}

bool Route::hasExcessLoad() const
{
    assert(!dirty);
    return excessLoad() > 0;
}

bool Route::hasExcessDistance() const
{
    assert(!dirty);
    return excessDistance() > 0;
}

bool Route::hasTimeWarp() const
{
#ifdef PYVRP_NO_TIME_WINDOWS
    return false;
#else
    assert(!dirty);
    return timeWarp() > 0;
#endif
}

size_t Route::idx() const { return idx_; }

Route::Node *Route::operator[](size_t idx)
{
    assert(idx < nodes.size());
    return nodes[idx];
}

template <typename... Segments>
Route::Proposal<Segments...> Route::proposal(Segments &&...segments) const
{
    return {this, data, std::forward<Segments>(segments)...};
}

Load Route::load() const
{
    assert(!dirty);
    return loadBefore.back().load();
}

Load Route::excessLoad() const
{
    assert(!dirty);
    return std::max<Load>(load() - capacity(), 0);
}

Distance Route::excessDistance() const
{
    assert(!dirty);
    return std::max<Distance>(distance() - maxDistance(), 0);
}

Load Route::capacity() const { return vehicleType_.capacity; }

size_t Route::startDepot() const { return vehicleType_.startDepot; }

size_t Route::endDepot() const { return vehicleType_.endDepot; }

Cost Route::fixedVehicleCost() const { return vehicleType_.fixedCost; }

Distance Route::distance() const
{
    assert(!dirty);
    return distBefore.back().distance();
}

Cost Route::distanceCost() const
{
    assert(!dirty);
    return unitDistanceCost() * static_cast<Cost>(distance());
}

Cost Route::unitDistanceCost() const { return vehicleType_.unitDistanceCost; }

Duration Route::duration() const
{
    assert(!dirty);
    return durBefore.back().duration();
}

Cost Route::durationCost() const
{
    assert(!dirty);
    return unitDurationCost() * static_cast<Cost>(duration());
}

Cost Route::unitDurationCost() const { return vehicleType_.unitDurationCost; }

Duration Route::maxDuration() const { return vehicleType_.maxDuration; }

Distance Route::maxDistance() const { return vehicleType_.maxDistance; }

Duration Route::timeWarp() const
{
    assert(!dirty);
    return durBefore.back().timeWarp(maxDuration());
}

size_t Route::profile() const { return vehicleType_.profile; }

bool Route::empty() const { return size() == 0; }

size_t Route::size() const
{
    assert(nodes.size() >= 2);  // excl. depots
    return nodes.size() - 2;
}

Route::SegmentAt Route::at(size_t idx) const
{
    assert(!dirty);
    return SegmentAt(*this, idx);
}

Route::SegmentAfter Route::after(size_t start) const
{
    assert(!dirty);
    return SegmentAfter(*this, start);
}

Route::SegmentBefore Route::before(size_t end) const
{
    assert(!dirty);
    return SegmentBefore(*this, end);
}

Route::SegmentBetween Route::between(size_t start, size_t end) const
{
    assert(!dirty);
    return SegmentBetween(*this, start, end);
}

template <typename... Segments>
Route::Proposal<Segments...>::Proposal(Route const *current,
                                       ProblemData const &data,
                                       Segments &&...segments)
    : current(current),
      data(data),
      segments(std::forward<Segments>(segments)...)
{
}

template <typename... Segments>
Route const *Route::Proposal<Segments...>::route() const
{
    return current;
}

template <typename... Segments>
DistanceSegment Route::Proposal<Segments...>::distanceSegment() const
{
    return std::apply(
        [&](auto &&...args)
        {
            return DistanceSegment::merge(
                data.distanceMatrix(current->profile()),
                args.distance(current->profile())...);
        },
        segments);
}

template <typename... Segments>
DurationSegment Route::Proposal<Segments...>::durationSegment() const
{
    return std::apply(
        [&](auto &&...args)
        {
            return DurationSegment::merge(
                data.durationMatrix(current->profile()),
                args.duration(current->profile())...);
        },
        segments);
}

template <typename... Segments>
LoadSegment Route::Proposal<Segments...>::loadSegment() const
{
    return std::apply([](auto &&...args)
                      { return LoadSegment::merge(args.load()...); },
                      segments);
}

}  // namespace pyvrp::search

// Outputs a route into a given ostream in CVRPLib format
std::ostream &operator<<(std::ostream &out, pyvrp::search::Route const &route);

#endif  // PYVRP_ROUTE_H
