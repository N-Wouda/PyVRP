.. figure:: assets/images/logo.svg
   :alt: PyVRP logo
   :figwidth: 100%

PyVRP is an open-source, state-of-the-art vehicle routing problem (VRP) solver.
It currently supports VRPs with:

* Pickups and deliveries between depots and clients (capacitated VRP, VRP with simultaneous pickup and delivery, VRP with backhaul);
* Vehicles of different capacities, costs, shift durations, routing profiles, and maximum distance and duration constraints (heterogeneous fleet VRP, site-dependent VRP);
* Time windows, client service durations, and release times (VRP with time windows and release times);
* Multiple depots (multi-depot VRP);
* Optional clients with prizes for visiting (prize collecting, team orienteering problem);
* Client groups imposing additional restrictions on multiple clients jointly (generalised VRP, VRP with multiple time windows).

The PyVRP package comes with pre-compiled binaries for Windows, Mac OS and Linux, and can thus be easily installed without requiring local compilation.
It can be installed through *pip* via

.. code-block:: shell

   pip install pyvrp

.. hint::

    If you are new to vehicle routing or metaheuristics, you might benefit from first reading the :doc:`introduction to VRP <setup/introduction_to_vrp>`, :doc:`introduction to HGS <setup/introduction_to_hgs>`, and :doc:`tutorial <../examples/quick_tutorial>` pages.
    To set up an installation from source, or to run the examples listed below yourself, please have a look at the :doc:`installation instructions <setup/installation>`.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   setup/introduction_to_vrp
   setup/introduction_to_hgs
   setup/installation
   setup/getting_help
   setup/benchmarks
   setup/citing

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/quick_tutorial
   examples/basic_vrps
   examples/using_pyvrp_components

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/pyvrp
   api/crossover
   api/diversity
   api/repair
   api/search
   api/plotting
   api/stop

.. toctree::
   :maxdepth: 1
   :caption: Developing PyVRP

   dev/benchmarking
   dev/contributing
   dev/supported_vrplib_fields
   dev/new_vrp_variants
