#include "Individual.h"
#include "ProblemData.h"

#include <fstream>
#include <numeric>
#include <vector>

using Client = int;
using Route = std::vector<Client>;
using Routes = std::vector<Route>;
using RouteType = int;

void Individual::evaluate(ProblemData const &data)
{
    numNonEmptyRoutes_ = 0;
    distance_ = 0;
    excessLoad_ = 0;
    timeWarp_ = 0;

    for (size_t idx = 0; idx < data.maxNumRoutes(); idx++)
    {
        auto const &route = routes_[idx];
        auto const &routeData = data.routeData(idx);
        if (route.empty())
            continue;

        numNonEmptyRoutes_++;

        int routeDist = data.dist(0, route[0]);
        int routeTimeWarp = 0;
        int routeLoad = data.client(route[0]).demand;

        int time = routeDist;

        if (time < data.client(route[0]).twEarly)
            time = data.client(route[0]).twEarly;

        if (time > data.client(route[0]).twLate)
        {
            routeTimeWarp += time - data.client(route[0]).twLate;
            time = data.client(route[0]).twLate;
        }

        for (size_t idx = 1; idx < route.size(); idx++)
        {
            routeDist += data.dist(route[idx - 1], route[idx]);
            routeLoad += data.client(route[idx]).demand;

            time += data.client(route[idx - 1]).serviceDuration
                    + data.dist(route[idx - 1], route[idx]);

            // Add possible waiting time
            if (time < data.client(route[idx]).twEarly)
                time = data.client(route[idx]).twEarly;

            // Add possible time warp
            if (time > data.client(route[idx]).twLate)
            {
                routeTimeWarp += time - data.client(route[idx]).twLate;
                time = data.client(route[idx]).twLate;
            }
        }

        // For the last client, the successors is the depot. Also update the
        // rDist and time
        routeDist += data.dist(route.back(), 0);
        time += data.client(route.back()).serviceDuration
                + data.dist(route.back(), 0);

        // For the depot, we only need to check the end of the time window
        // (add possible time warp)
        routeTimeWarp += std::max(time - data.depot().twLate, 0);

        // Whole solution stats
        distance_ += routeDist;
        timeWarp_ += routeTimeWarp;

        if (static_cast<size_t>(routeLoad) > routeData.vehicleCapacity)
            excessLoad_ += routeLoad - routeData.vehicleCapacity;
    }
}

size_t Individual::numNonEmptyRoutes() const { return numNonEmptyRoutes_; }

Routes const &Individual::getRoutes() const { return routes_; }

std::vector<std::pair<Client, Client>> const &Individual::getNeighbours() const
{
    return neighbours;
}

std::vector<RouteType> const &Individual::getAssignments() const
{
    return assignedRouteTypes;
}

bool Individual::isFeasible() const
{
    return !hasExcessLoad() && !hasTimeWarp();
}

bool Individual::hasExcessLoad() const { return excessLoad_ > 0; }

bool Individual::hasTimeWarp() const { return timeWarp_ > 0; }

size_t Individual::distance() const { return distance_; }

size_t Individual::excessLoad() const { return excessLoad_; }

size_t Individual::timeWarp() const { return timeWarp_; }

void Individual::makeNeighbours(ProblemData const &data)
{
    neighbours[0] = {0, 0};  // note that depot neighbours have no meaning

    for (size_t rIdx = 0; rIdx != data.maxNumRoutes(); ++rIdx)
    {
        auto const route = routes_[rIdx];
        for (size_t idx = 0; idx != route.size(); ++idx)
            neighbours[route[idx]]
                = {idx == 0 ? 0 : route[idx - 1],                  // pred
                   idx == route.size() - 1 ? 0 : route[idx + 1]};  // succ
    }
}

void Individual::makeAssignedRouteTypes(ProblemData const &data)
{
    assignedRouteTypes[0] = -1;  // unassigned

    for (size_t rIdx = 0; rIdx != data.maxNumRoutes(); ++rIdx)
    {
        auto const route = routes_[rIdx];
        for (size_t idx = 0; idx != route.size(); ++idx)
            assignedRouteTypes[route[idx]] = data.routeType(rIdx);
    }
}

bool Individual::operator==(Individual const &other) const
{
    // First compare simple attributes, since that's a quick and cheap check.
    // Only when these are the same we test if the neighbours are all equal.
    // Only when that is also the case, we check if the assigned vehicle types
    // (capacities) are equal for the heterogeneous case.

    // clang-format off
    return distance_ == other.distance_
        && excessLoad_ == other.excessLoad_
        && timeWarp_ == other.timeWarp_
        && numNonEmptyRoutes_ == other.numNonEmptyRoutes_
        && neighbours == other.neighbours
        && assignedRouteTypes == other.assignedRouteTypes;
    // clang-format on
}

Individual::Individual(ProblemData const &data, XorShift128 &rng)
    : routes_(data.maxNumRoutes()),
      neighbours(data.numClients() + 1),
      assignedRouteTypes(data.numClients() + 1)
{
    // Shuffle clients (to create random routes)
    auto clients = std::vector<int>(data.numClients());
    std::iota(clients.begin(), clients.end(), 1);
    std::shuffle(clients.begin(), clients.end(), rng);

    // Distribute clients evenly over the routes: the total number of clients
    // per vehicle, with an adjustment in case the division is not perfect.
    auto const maxNumRoutes = data.maxNumRoutes();
    auto const numClients = data.numClients();
    auto const perRouteFloor = std::max(numClients / maxNumRoutes, size_t(1));
    auto const perRoute = perRouteFloor + (numClients % maxNumRoutes != 0);

    for (size_t idx = 0; idx != numClients; ++idx)
        routes_[idx / perRoute].push_back(clients[idx]);

    makeNeighbours(data);
    makeAssignedRouteTypes(data);
    evaluate(data);
}

Individual::Individual(ProblemData const &data, Routes routes)
    : routes_(std::move(routes)),
      neighbours(data.numClients() + 1),
      assignedRouteTypes(data.numClients() + 1)
{
    if (routes_.size() > data.maxNumRoutes())
    {
        auto const msg = "Number of routes must not exceed number of vehicles.";
        throw std::runtime_error(msg);
    }

    // Shift routes to the front as much as possible among routes of the same
    // type (vehicle capacity)
    size_t j = 0;  // Index of next position to put non-empty route
    for (size_t i = 0; i < routes_.size(); i++)
    {
        // In order to shift forward, routes must be exchangable
        // so the route data must be equal (same depot and capacity)
        // Note that this assumes exchangable routes are grouped together
        if (data.routeType(i) != data.routeType(j))
        {
            // Move to next group
            j = i;
        }
        // Note that it holds that i >= j
        if (!routes_[i].empty())
        {
            // Only shift the route if i != j (which means i > j)
            if (i != j)
            {
                routes_[j] = routes_[i];
                routes_[i].clear();
            }
            j++;
        }
    }

    // Expand to at least numVehicles routes, where any newly inserted routes
    // will be empty.
    routes_.resize(data.maxNumRoutes());

    makeNeighbours(data);
    makeAssignedRouteTypes(data);
    evaluate(data);
}

std::ostream &operator<<(std::ostream &out, Individual const &indiv)
{
    // Since non-empty routes are guaranteed to come before empty routes
    // this will print consecutive route numbers for homogeneous problem
    // instances, but there may be gaps in the route indices corresponding
    // to different vehicle capacities.
    auto const &routes = indiv.getRoutes();

    for (size_t rIdx = 0; rIdx != routes.size(); ++rIdx)
    {
        if (routes[rIdx].empty())
            continue;
        out << "Route #" << rIdx + 1 << ":";  // route number
        for (int cIdx : routes[rIdx])
            out << " " << cIdx;  // client index
        out << '\n';
    }

    out << "Distance: " << indiv.distance() << '\n';
    return out;
}
