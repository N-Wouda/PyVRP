#ifndef PYVRP_SUBPOPULATION_H
#define PYVRP_SUBPOPULATION_H

#include "CostEvaluator.h"
#include "Solution.h"
#include "diversity/diversity.h"

#include <functional>
#include <vector>

namespace pyvrp
{
/**
 * PopulationParams(
 *     min_pop_size: int = 25,
 *     generation_size: int = 40,
 *     nb_elite: int = 4,
 *     nb_close: int = 5,
 *     lb_diversity: float = 0.1,
 *     ub_diversity: float = 0.5,
 * )
 *
 * Parameter configuration for the :class:`~pyvrp.Population.Population`.
 *
 * Attributes
 * ----------
 * min_pop_size
 *     Minimum subpopulation size. This is the size of the subpopulation after
 *     survivor selection.
 * generation_size
 *     The size of a single generation, that is, the number of new solutions
 *     inserted into a subpopulation between survivor selections.
 * nb_elite
 *     Number of elite solutions. This number of fittest solutions are always
 *     survivors.
 * nb_close
 *     Number of close solutions. These are used to determine similarity
 *     between solutions, which is an important component of fitness.
 * lb_diversity
 *     A lower bound on the diversity of the solutions selected for
 *     tournament. See :meth:`~pyvrp.Population.Population.select` for details.
 * ub_diversity
 *     An upper bound on the diversity of the solutions selected for
 *     tournament. See :meth:`~pyvrp.Population.Population.select` for details.
 */
struct PopulationParams
{
    size_t const minPopSize;
    size_t const generationSize;
    size_t const nbElite;
    size_t const nbClose;
    double const lbDiversity;
    double const ubDiversity;

    PopulationParams(size_t minPopSize = 25,
                     size_t generationSize = 40,
                     size_t nbElite = 4,
                     size_t nbClose = 5,
                     double lbDiversity = 0.1,
                     double ubDiversity = 0.5);

    bool operator==(PopulationParams const &other) const = default;

    /**
     * Returns the maximum subpopulation size.
     */
    size_t maxPopSize() const;
};

/**
 * SubPopulation(diversity_op: DiversityMeasure, params: PopulationParams)
 *
 * Creates a SubPopulation instance.
 *
 * This subpopulation manages a collection of solutions, and initiates
 * survivor selection (purging) when their number grows large. A
 * subpopulation's solutions can be accessed via indexing and iteration.
 * Each solution is stored as a tuple of type ``_Item``, which stores
 * the solution itself, a fitness score (higher is worse), and a list
 * of proximity values to the other solutions in the subpopulation.
 *
 * Parameters
 * ----------
 * diversity_op
 *     Operator to use to determine pairwise diversity between solutions.
 * params
 *     Population parameters.
 */
class SubPopulation
{
    diversity::DiversityMeasure divOp;
    PopulationParams const &params;  // owned by Population, on the Python side

public:
    struct Item
    {
        using Proximity = std::vector<std::pair<double, Solution const *>>;

        PopulationParams const *params;

        // Note that this pointer is not owned by the Item - it is merely a
        // reference to memory owned and allocated by the SubPopulation this
        // item is part of. The SubPopulation remains responsible for managing
        // that memory.
        Solution const *solution;

        // Fitness should be used carefully: only directly after updateFitness
        // was called. At any other moment, it will be outdated.
        double fitness;
        Proximity proximity;

        double avgDistanceClosest() const;
    };

private:
    std::vector<Item> items;

    // Removes the element at the given iterator location from the items.
    void remove(std::vector<Item>::iterator const &iterator);

public:
    SubPopulation(diversity::DiversityMeasure divOp,
                  PopulationParams const &params);

    ~SubPopulation();

    /**
     * Adds the given solution to the subpopulation. Survivor selection is
     * automatically triggered when the population reaches its maximum size.
     *
     * Parameters
     * ----------
     * solution
     *     Solution to add to the subpopulation.
     * cost_evaluator
     *     CostEvaluator to use to compute the cost.
     */
    void add(Solution const *solution, CostEvaluator const &costEvaluator);

    std::vector<Item>::const_iterator cbegin() const;

    std::vector<Item>::const_iterator cend() const;

    size_t size() const;

    Item const &operator[](size_t idx) const;

    /**
     * Performs survivor selection: solutions in the subpopulation are
     * purged until the population is reduced to the ``min_pop_size``.
     * Purging happens to duplicate solutions first, and then to solutions
     * with high biased fitness.
     *
     * Parameters
     * ----------
     * cost_evaluator
     *     CostEvaluator to use to compute the cost.
     */
    void purge(CostEvaluator const &costEvaluator);

    /**
     * Updates the biased fitness scores of solutions in the subpopulation.
     * This fitness depends on the quality of the solution (based on its cost)
     * and the diversity w.r.t. to other solutions in the subpopulation.
     *
     * .. warning::
     *
     *    This function must be called before accessing the
     *    :meth:`~SubPopulationItem.fitness` attribute.
     */
    void updateFitness(CostEvaluator const &costEvaluator);
};
}  // namespace pyvrp

#endif  // PYVRP_SUBPOPULATION_H
