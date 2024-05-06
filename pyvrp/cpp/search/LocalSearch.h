#ifndef PYVRP_LOCALSEARCH_H
#define PYVRP_LOCALSEARCH_H

#include "CostEvaluator.h"
#include "LocalSearchOperator.h"
#include "ProblemData.h"
#include "RandomNumberGenerator.h"
#include "Route.h"
#include "Solution.h"

#include <functional>
#include <stdexcept>
#include <vector>

namespace pyvrp::search
{
class LocalSearch
{
    using NodeOp = LocalSearchOperator<Route::Node>;
    using RouteOp = LocalSearchOperator<Route>;
    using Neighbours = std::vector<std::vector<size_t>>;

    ProblemData const &data;

    // Neighborhood restrictions: list of nearby clients for each client (size
    // numLocations, but nothing is stored for the depots!)
    Neighbours neighbours_;

    std::vector<size_t> orderNodes;   // node order used by LS::search
    std::vector<size_t> orderRoutes;  // route order used by LS::intensify

    std::vector<int> lastModified;  // tracks when routes were last modified

    std::vector<Route::Node> nodes;
    std::vector<Route> routes;

    std::vector<NodeOp *> nodeOps;
    std::vector<RouteOp *> routeOps;

    int numMoves = 0;              // Operator counter
    bool searchCompleted = false;  // No further improving move found?

    // Load an initial solution that we will attempt to improve.
    void loadSolution(Solution const &solution);

    // Export the LS solution back into a solution.
    Solution exportSolution() const;

    // Tests the node pair (U, V).
    bool applyNodeOps(Route::Node *U,
                      Route::Node *V,
                      CostEvaluator const &costEvaluator);

    // Tests the route pair (U, V).
    bool applyRouteOps(Route *U, Route *V, CostEvaluator const &costEvaluator);

    // Tests moves involving empty routes.
    void applyEmptyRouteMoves(Route::Node *U,
                              CostEvaluator const &costEvaluator);

    // Tests moves involving missing or optional clients.
    void applyOptionalClientMoves(Route::Node *U,
                                  CostEvaluator const &costEvaluator);

    // Tests moves involving clients in client groups.
    void applyGroupMoves(Route::Node *U, CostEvaluator const &costEvaluator);

    // Updates solution state after an improving local search move.
    void update(Route *U, Route *V);

    // Performs search on the currently loaded solution.
    void search(CostEvaluator const &costEvaluator);

    // Performs intensify on the currently loaded solution.
    void intensify(CostEvaluator const &costEvaluator,
                   double overlapTolerance = 0.05);

    // Evaluate and apply inserting U after one of its neighbours if it's an
    // improving move or required for feasibility.
    void
    insert(Route::Node *U, CostEvaluator const &costEvaluator, bool required);

public:
    /**
     * Adds a local search operator that works on node/client pairs U and V.
     */
    void addNodeOperator(NodeOp &op);

    /**
     * Adds a local search operator that works on route pairs U and V.
     */
    void addRouteOperator(RouteOp &op);

    /**
     * Set neighbourhood structure to use by the local search. For each client,
     * the neighbourhood structure is a vector of nearby clients. Depots have
     * no nearby client.
     */
    void setNeighbours(Neighbours neighbours);

    /**
     * @return The neighbourhood structure currently in use.
     */
    Neighbours const &neighbours() const;

    /**
     * Iteratively calls ``search()`` and ``intensify()`` until no further
     * improvements are made.
     */
    Solution operator()(Solution const &solution,
                        CostEvaluator const &costEvaluator);

    /**
     * Performs regular (node-based) local search around the given solution,
     * and returns a new, hopefully improved solution.
     */
    Solution search(Solution const &solution,
                    CostEvaluator const &costEvaluator);

    /**
     * Performs a more intensive route-based local search around the given
     * solution, and returns a new, hopefully improved solution.
     */
    Solution intensify(Solution const &solution,
                       CostEvaluator const &costEvaluator,
                       double overlapTolerance = 0.05);

    /**
     * Shuffles the order in which the node and route pairs are evaluated, and
     * the order in which node and route operators are applied.
     */
    void shuffle(RandomNumberGenerator &rng);

    LocalSearch(ProblemData const &data, Neighbours neighbours);
};
}  // namespace pyvrp::search

#endif  // PYVRP_LOCALSEARCH_H
