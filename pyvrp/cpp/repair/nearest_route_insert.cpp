#include "nearest_route_insert.h"
#include "repair.h"

#include "TimeWindowSegment.h"
#include "search/primitives.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using pyvrp::Solution;
using pyvrp::search::insertCost;
using pyvrp::search::Route;

using Clients = std::vector<Route::Node>;
using Routes = std::vector<Route>;
using SolRoutes = std::vector<Solution::Route>;

Solution pyvrp::repair::nearestRouteInsert(SolRoutes const &solRoutes,
                                           std::vector<size_t> const &unplanned,
                                           ProblemData const &data,
                                           CostEvaluator const &costEvaluator)
{
    if (solRoutes.empty() && !unplanned.empty())
        throw std::invalid_argument("Need routes to repair!");

    Clients clients;
    Routes routes;
    setupRoutes(clients, routes, solRoutes, data);

    for (size_t client : unplanned)
    {
        Route::Node *U = &clients[client];
        assert(!U->route());

        auto const x = static_cast<double>(data.client(client).x);
        auto const y = static_cast<double>(data.client(client).y);

        // Determine route with centroid nearest to this client.
        auto const cmp = [&](auto const &a, auto const &b) {
            if (a.empty())
                return false;

            if (b.empty())
                return true;

            auto const [aX, aY] = a.centroid();
            auto const [bX, bY] = b.centroid();
            return std::hypot(x - aX, y - aY) < std::hypot(x - bX, y - bY);
        };

        auto &route = *std::max_element(routes.begin(), routes.end(), cmp);

        // Find best insertion point in selected route.
        Cost bestCost = insertCost(U, route[0], data, costEvaluator);
        auto offset = 0;

        for (auto *V : route)  // evaluate after V
        {
            auto const cost = insertCost(U, V, data, costEvaluator);
            if (cost < bestCost)
            {
                bestCost = cost;
                offset = V->idx() + 1;
            }
        }

        route.insert(offset, U);
        route.update();
    }

    return exportRoutes(data, routes);
}
