#include "bindings.h"
#include "Exchange.h"
#include "LocalSearch.h"
#include "Route.h"
#include "SwapRoutes.h"
#include "SwapStar.h"
#include "SwapTails.h"
#include "primitives.h"
#include "search_docs.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;

using pyvrp::search::Exchange;
using pyvrp::search::inplaceCost;
using pyvrp::search::insertCost;
using pyvrp::search::LocalSearch;
using pyvrp::search::LocalSearchOperator;
using pyvrp::search::removeCost;
using pyvrp::search::Route;
using pyvrp::search::SwapRoutes;
using pyvrp::search::SwapStar;
using pyvrp::search::SwapTails;

PYBIND11_MODULE(_search, m)
{
    using NodeOp = LocalSearchOperator<pyvrp::search::Route::Node>;
    using RouteOp = LocalSearchOperator<pyvrp::search::Route>;

    py::class_<NodeOp>(m, "NodeOperator");
    py::class_<RouteOp>(m, "RouteOperator");

    py::class_<Exchange<1, 0>, NodeOp>(
        m, "Exchange10", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<1, 0>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<1, 0>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<2, 0>, NodeOp>(
        m, "Exchange20", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<2, 0>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<2, 0>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<3, 0>, NodeOp>(
        m, "Exchange30", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<3, 0>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<3, 0>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<1, 1>, NodeOp>(
        m, "Exchange11", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<1, 1>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<1, 1>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<2, 1>, NodeOp>(
        m, "Exchange21", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<2, 1>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<2, 1>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<3, 1>, NodeOp>(
        m, "Exchange31", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<3, 1>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<3, 1>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<2, 2>, NodeOp>(
        m, "Exchange22", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<2, 2>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<2, 2>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<3, 2>, NodeOp>(
        m, "Exchange32", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<3, 2>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<3, 2>::apply, py::arg("U"), py::arg("V"));

    py::class_<Exchange<3, 3>, NodeOp>(
        m, "Exchange33", DOC(pyvrp, search, Exchange))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &Exchange<3, 3>::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &Exchange<3, 3>::apply, py::arg("U"), py::arg("V"));

    py::class_<SwapRoutes, RouteOp>(
        m, "SwapRoutes", DOC(pyvrp, search, SwapRoutes))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &SwapRoutes::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &SwapRoutes::apply, py::arg("U"), py::arg("V"));

    py::class_<SwapStar, RouteOp>(m, "SwapStar", DOC(pyvrp, search, SwapStar))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &SwapStar::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &SwapStar::apply, py::arg("U"), py::arg("V"));

    py::class_<SwapTails, NodeOp>(m, "SwapTails", DOC(pyvrp, search, SwapTails))
        .def(py::init<pyvrp::ProblemData const &>(),
             py::arg("data"),
             py::keep_alive<1, 2>())  // keep data alive
        .def("evaluate",
             &SwapTails::evaluate,
             py::arg("U"),
             py::arg("V"),
             py::arg("cost_evaluator"))
        .def("apply", &SwapTails::apply, py::arg("U"), py::arg("V"));

    py::class_<LocalSearch>(m, "LocalSearch")
        .def(py::init<pyvrp::ProblemData const &,
                      std::vector<std::vector<size_t>>>(),
             py::arg("data"),
             py::arg("neighbours"),
             py::keep_alive<1, 2>())  // keep data alive until LS is freed
        .def("add_node_operator",
             &LocalSearch::addNodeOperator,
             py::arg("op"),
             py::keep_alive<1, 2>())
        .def("add_route_operator",
             &LocalSearch::addRouteOperator,
             py::arg("op"),
             py::keep_alive<1, 2>())
        .def("set_neighbours",
             &LocalSearch::setNeighbours,
             py::arg("neighbours"))
        .def("neighbours",
             &LocalSearch::neighbours,
             py::return_value_policy::reference_internal)
        .def("__call__",
             &LocalSearch::operator(),
             py::arg("solution"),
             py::arg("cost_evaluator"))
        .def("search",
             py::overload_cast<pyvrp::Solution const &,
                               pyvrp::CostEvaluator const &>(
                 &LocalSearch::search),
             py::arg("solution"),
             py::arg("cost_evaluator"))
        .def("intensify",
             py::overload_cast<pyvrp::Solution const &,
                               pyvrp::CostEvaluator const &,
                               double const>(&LocalSearch::intensify),
             py::arg("solution"),
             py::arg("cost_evaluator"),
             py::arg("overlap_tolerance") = 0.05)
        .def("shuffle", &LocalSearch::shuffle, py::arg("rng"));

    py::class_<Route>(m, "Route", DOC(pyvrp, search, Route))
        .def(py::init<pyvrp::ProblemData const &, size_t, size_t>(),
             py::arg("data"),
             py::arg("idx"),
             py::arg("vehicle_type"),
             py::keep_alive<1, 2>())  // keep data alive
        .def_property_readonly("idx", &Route::idx)
        .def_property_readonly("vehicle_type", &Route::vehicleType)
        .def("__delitem__", &Route::remove, py::arg("idx"))
        .def("__getitem__",
             &Route::operator[],
             py::arg("idx"),
             py::return_value_policy::reference_internal)
        .def(
            "__iter__",
            [](Route const &route)
            { return py::make_iterator(route.begin(), route.end()); },
            py::return_value_policy::reference_internal)
        .def("__len__", &Route::size)
        .def("__str__",
             [](Route const &route)
             {
                 std::stringstream stream;
                 stream << route;
                 return stream.str();
             })
        .def("is_feasible", &Route::isFeasible)
        .def("has_excess_load", &Route::hasExcessLoad)
        .def("has_excess_distance", &Route::hasExcessDistance)
        .def("has_time_warp", &Route::hasTimeWarp)
        .def("capacity", &Route::capacity)
        .def("depot", &Route::depot)
        .def("fixed_vehicle_cost", &Route::fixedVehicleCost)
        .def("load", &Route::load)
        .def("excess_load", &Route::excessLoad)
        .def("excess_distance", &Route::excessDistance)
        .def("distance", &Route::distance)
        .def("distance_cost", &Route::distanceCost)
        .def("unit_distance_cost", &Route::unitDistanceCost)
        .def("duration", &Route::duration)
        .def("duration_cost", &Route::durationCost)
        .def("unit_duration_cost", &Route::unitDurationCost)
        .def("max_duration", &Route::maxDuration)
        .def("max_distance", &Route::maxDistance)
        .def("time_warp", &Route::timeWarp)
        .def("profile", &Route::profile)
        .def(
            "dist_at",
            [](Route const &route, size_t idx)
            { return static_cast<pyvrp::DistanceSegment>(route.at(idx)); },
            py::arg("idx"))
        .def(
            "dist_between",
            [](Route const &route, size_t start, size_t end) {
                return static_cast<pyvrp::DistanceSegment>(
                    route.between(start, end));
            },
            py::arg("start"),
            py::arg("end"))
        .def(
            "dist_after",
            [](Route const &route, size_t start)
            { return static_cast<pyvrp::DistanceSegment>(route.after(start)); },
            py::arg("start"))
        .def(
            "dist_before",
            [](Route const &route, size_t end)
            { return static_cast<pyvrp::DistanceSegment>(route.before(end)); },
            py::arg("end"))
        .def(
            "load_at",
            [](Route const &route, size_t idx)
            { return static_cast<pyvrp::LoadSegment>(route.at(idx)); },
            py::arg("idx"))
        .def(
            "load_between",
            [](Route const &route, size_t start, size_t end) {
                return static_cast<pyvrp::LoadSegment>(
                    route.between(start, end));
            },
            py::arg("start"),
            py::arg("end"))
        .def(
            "load_after",
            [](Route const &route, size_t start)
            { return static_cast<pyvrp::LoadSegment>(route.after(start)); },
            py::arg("start"))
        .def(
            "load_before",
            [](Route const &route, size_t end)
            { return static_cast<pyvrp::LoadSegment>(route.before(end)); },
            py::arg("end"))
        .def(
            "duration_at",
            [](Route const &route, size_t idx)
            { return static_cast<pyvrp::DurationSegment>(route.at(idx)); },
            py::arg("idx"))
        .def(
            "duration_between",
            [](Route const &route, size_t start, size_t end) {
                return static_cast<pyvrp::DurationSegment>(
                    route.between(start, end));
            },
            py::arg("start"),
            py::arg("end"))
        .def(
            "duration_after",
            [](Route const &route, size_t start)
            { return static_cast<pyvrp::DurationSegment>(route.after(start)); },
            py::arg("start"))
        .def(
            "duration_before",
            [](Route const &route, size_t end)
            { return static_cast<pyvrp::DurationSegment>(route.before(end)); },
            py::arg("end"))
        .def("centroid", &Route::centroid)
        .def("overlaps_with",
             &Route::overlapsWith,
             py::arg("other"),
             py::arg("tolerance"))
        .def("append",
             &Route::push_back,
             py::arg("node"),
             py::keep_alive<1, 2>(),  // keep node alive
             py::keep_alive<2, 1>())  // keep route alive
        .def("clear", &Route::clear)
        .def("insert",
             &Route::insert,
             py::arg("idx"),
             py::arg("node"),
             py::keep_alive<1, 3>(),  // keep node alive
             py::keep_alive<3, 1>())  // keep route alive
        .def("update", &Route::update);

    py::class_<Route::Node>(m, "Node", DOC(pyvrp, search, Route, Node))
        .def(py::init<size_t>(), py::arg("loc"))
        .def_property_readonly("client", &Route::Node::client)
        .def_property_readonly("idx", &Route::Node::idx)
        .def_property_readonly("route", &Route::Node::route)
        .def("is_depot", &Route::Node::isDepot);

    m.def("insert_cost",
          &insertCost,
          py::arg("U"),
          py::arg("V"),
          py::arg("data"),
          py::arg("cost_evaluator"),
          DOC(pyvrp, search, insertCost));

    m.def("inplace_cost",
          &inplaceCost,
          py::arg("U"),
          py::arg("V"),
          py::arg("data"),
          py::arg("cost_evaluator"),
          DOC(pyvrp, search, inplaceCost));

    m.def("remove_cost",
          &removeCost,
          py::arg("U"),
          py::arg("data"),
          py::arg("cost_evaluator"),
          DOC(pyvrp, search, removeCost));
}
