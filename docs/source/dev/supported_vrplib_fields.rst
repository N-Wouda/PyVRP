The VRPLIB format
=================

PyVRP supports reading benchmark instances in the VRPLIB format through the `VRPLIB <https://github.com/PyVRP/VRPLIB>`_ package.
Although generally used, this format is not all that well standardised, so it merits some description what PyVRP actually supports.
In addition to standard specifications and data sections, PyVRP uses several data sections that are not commonly found in (other) benchmark instances.
On this page, we explain all specifications and data sections that PyVRP understands.

This page can be useful in understanding how benchmark instances should be formatted to work with PyVRP's :meth:`~pyvrp.read.read` function.
All instances in our `instances repository <https://github.com/PyVRP/Instances>`_ adhere to this format, and use the specifications and data sections described below.

.. note::

   Any specification or data section in a VRPLIB-formatted file that is *not* defined in the glossaries below is not understood by PyVRP, and will be silently ignored.


Specifications
--------------

Specifications are key-value pairs.
PyVRP supports the following specifications:

.. glossary::
   :sorted:

   ``CAPACITY``
      Homogeneous capacity of all vehicles in the instance.
      To specify heterogeneous capacities, see :term:`CAPACITY_SECTION`.

   ``DIMENSION``
      Number of locations in the instance.
      This is the sum of the number of depots and the number of clients.

   ``EDGE_WEIGHT_FORMAT``
      Specifies the format of the :term:`EDGE_WEIGHT_SECTION` if the edge weights are given explicitly.

   ``EDGE_WEIGHT_TYPE``
      Specifies how the edge weights (distances and durations) are given.

   ``SERVICE_TIME``
      Service time for all clients.
      To specify heterogeneous client service times, see :term:`SERVICE_TIME_SECTION`.

   ``VEHICLES``
      Number of vehicles in the instance.
      The number of vehicles defaults to the number of clients in the instance when this specification is not provided: PyVRP assumes an unlimited fleet in this case.

   ``VEHICLES_MAX_DISTANCE``
      Maximum route distance for all vehicles.
      To specify heterogeneous maximum route distances, see :term:`VEHICLES_MAX_DISTANCE_SECTION`.

   ``VEHICLES_MAX_DURATION``
      Maximum route duration for all vehicles.
      To specify heterogeneous maximum route durations, see :term:`VEHICLES_MAX_DURATION_SECTION`.

Data sections
-------------

Data sections are array-like values that specify the actual problem data.
PyVRP supports the following data sections:

.. glossary::
   :sorted:

   ``BACKHAUL_SECTION``
      Array of backhaul quantities, one for each location.
      This is the amount picked up at the client and transported back to the depot.

   ``CAPACITY_SECTION``
      Capacity for each vehicle, typically used in heterogeneous fleet instances.
      Vehicle capacities are assumed to be unconstrained if this section is not present.

   ``DEMAND_SECTION``
   ``LINEHAUL_SECTION``
      Array of demands, one for each location.
      This is the delivery amount from the depot to the client.

   ``DEPOT_SECTION``
      Array of location indices that are depots.
      These location indices should be the contiguous lower indices, starting from 1.

   ``EDGE_WEIGHT_SECTION``
      When provided, this section explicitly describes the distance and duration matrices.
      If not provided, such matrices are computed based on what's specified for :term:`EDGE_WEIGHT_FORMAT` and :term:`EDGE_WEIGHT_TYPE`.

   ``MUTUALLY_EXCLUSIVE_GROUP_SECTION``
      When provided, this section describes client memberships of mutually exclusive groups.
      Of all clients in such a group, exactly one must be visited.    

   ``NODE_COORD_SECTION``
      Array of :math:`(x, y)` coordinates for each location.

   ``PRIZE_SECTION``
      Array of prizes for visiting each location.
      A value of zero for non-depots implies visiting that location is required.

   ``RELEASE_TIME_SECTION``
      Array of release times for each location.

   ``SERVICE_TIME_SECTION``
      Array of service durations for each location.

   ``TIME_WINDOW_SECTION``
      Array of :math:`[e, l]` time window data, for each location.

   ``VEHICLES_ALLOWED_CLIENTS_SECTION``
      Lists specifying the client locations that each vehicle is allowed to service, one list for each vehicle.
      Each vehicle is allowed to visit each client if this section is not provided.
      This section is commonly used in site-dependent instances.

   ``VEHICLES_DEPOT_SECTION``
      Depot assignments for each vehicle, typically used in multi-depot instances.
      Vehicles are assigned to the first depot if this section is not provided.

   ``VEHICLES_MAX_DISTANCE_SECTION``
      Maximum route distance for each vehicle.
      Route distances are assumed to be unconstrained if this section is not present.

   ``VEHICLES_MAX_DURATION_SECTION``
      Maximum route duration for each vehicle.
      Route durations are assumed to be unconstrained if this section is not present.
