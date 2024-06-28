Benchmarks
==========

This page lists benchmark results obtained by recent versions of the PyVRP package and reference VRP solvers.

.. hint::

   See the :doc:`benchmarking instructions <../dev/benchmarking>` for details on how we benchmark. 

PyVRP
-----

The table below contains the benchmark results obtained by each new significant version of PyVRP.
The reported values represent the percentage gap to the best-known solutions at the time of the benchmark, averaged over ten seeds.
These best-known solutions are tracked in the `PyVRP/Instances <https://github.com/PyVRP/Instances>`_ repository.

.. note::

   We always benchmark against the most recent best-known solutions.

.. list-table::
   :header-rows: 1

   * - Date
     - Version
     - CVRP
     - VRPTW
     - PCVRPTW
     - MDVRPTW
     - VRPB
   * - 27 June 2024
     - `0.9.0 <https://github.com/PyVRP/PyVRP/tree/e74c9b19915c308aae6374fab8507a0f0e48ec72>`_
     - 0.24%
     - 0.61%
     - 0.41%
     - 1.01%
     - 0.40%
   * - 22 March 2024
     - `0.8.0 <https://github.com/PyVRP/PyVRP/tree/75e4fd4f5a449f11d4974164ce84a170a53b8221>`_
     - 0.23%
     - 0.58%
     - 0.39%
     - 0.99% [#note1]_
     - 0.35%
   * - 29 January 2024
     - `0.7.0 <https://github.com/PyVRP/PyVRP/tree/c3e685a7bd5e028322c19f5c83ed9c935ccdae8e>`_
     - 0.23%
     - 0.48%
     - 0.21% [#note2]_
     - 0.44%
     -
   * - 31 August 2023
     - `0.6.0 <https://github.com/PyVRP/PyVRP/tree/7ce7bfe66cb4930496dab412eb0f1999b18fbfa8>`_
     - 0.24%
     - 0.54%
     -
     -
     -
   * - 1 August 2023
     - `0.5.0 <https://github.com/PyVRP/PyVRP/tree/d4799a810a8cf7d16ea2c8871204bdfb3a896d06>`_
     - 0.22%
     - 0.40%
     -
     -
     -
   * - 9 July 2023
     - `0.4.2 <https://github.com/PyVRP/PyVRP/tree/f934e0da184dd0bdbd4d83f72ec98b7ef51cd8da>`_
     - 0.18%
     - 0.43%
     -
     -
     -
   * - 20 May 2023
     - `0.3.0 <https://github.com/PyVRP/PyVRP/tree/4632ce97cedbc9d58216c2bec43cd679eb1d21c9>`_
     - 0.25%
     - 0.43%
     -
     -
     -
   * - 13 May 2023
     - `0.2.0 <https://github.com/PyVRP/PyVRP/tree/3784f03fa3b6777613fb0bc8cedeac5ad372cfe4>`_
     - 0.25%
     - 0.43%
     -
     -
     -
   * - 28 February 2023
     - `0.1.0 <https://github.com/PyVRP/PyVRP/tree/e1b1ac72bc1246cc51d252bf72df71fc43dc422b>`_
     - 0.22%
     - 0.45%
     -
     -
     -


Reference VRP solvers
---------------------

The table below contains the benchmark results obtained by reference VRP solvers.
The reported values represent the average gaps to the best-known solutions (at the time of the benchmark) over ten seeds.

.. list-table::
   :header-rows: 1

   * - Date
     - Name
     - CVRP
     - VRPTW
     - PCVRPTW
     - MDVRPTW
     - VRPB
   * - 20 May 2024
     - `Google OR-Tools <https://developers.google.com/optimization/routing>`_ [#note3]_
     - 4.42%
     - 9.38%
     - 10.72%
     -
     -
   * - 28 February 2023
     - `HGS-CVRP <https://github.com/vidalt/HGS-CVRP>`_
     - 0.11%
     -
     -
     -
     -
   * - 28 February 2023
     - `HGS-DIMACS <https://github.com/ortec/euro-neurips-vrp-2022-quickstart#baseline-solver-hybrid-genetic-search-hgs>`_
     -
     - 0.32%
     -
     -
     -
   * - 2020
     - `ILS-SP <https://doi.org/10.1007/s11590-020-01564-5>`_
     -
     -
     -
     -
     - 1.09% [#note4]_
   * - 2013
     - `HGS-ADC <https://doi.org/10.1016/j.cor.2012.07.018>`_
     -
     -
     -
     - 0.71% [#note5]_
     -


.. rubric:: Notes

.. [#note1]
   In v0.8.0, we have simplified the local search algorithm to contain fewer operators, causing the "no-improvement" stopping criterion to be reached much more quickly.
   This means that the current MDVRPTW benchmark runs much shorter compared to the previous versions.
   This result should therefore not be perceived as a significant change in performance.

.. [#note2]
   PyVRP supports prize-collecting since v0.3.0, but due to a parsing error results from versions before v0.7.0 were incorrect.
   We have corrected the issue in PyVRP, and report gaps on versions from v0.7.0 onwards.
   The best-known solutions have been updated to address this issue.

.. [#note3]
   Results obtained using Google OR-Tools `v9.10.4067 <https://pypi.org/project/ortools/9.10.4067/>`_.
   The reported average gap is over only one seed, because there is no interface to set the seed.
   The code used to benchmark Google OR-tools can be found in `this issue <https://github.com/PyVRP/PyVRP/issues/453>`_.

.. [#note4]
   Literature result from Appendix B of `Subramanian and Queiroga (2020) <https://doi.org/10.1007/s11590-020-01564-5>`_.
   We use the result from the best performing variant of ILS-SP, that is, :math:`ILS_{B}-SP_{B}`.

.. [#note5]
   Literature result from Table 8 of `Vidal et al. (2013) <https://doi.org/10.1016/j.cor.2012.07.018>`_.
   This is an average gap over five seeds, rather than the usual ten.
   Note that this paper introduced the MDVRPTW benchmark instances.
