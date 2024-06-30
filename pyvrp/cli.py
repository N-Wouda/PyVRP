import argparse
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.contrib.concurrent import process_map

from pyvrp import ProblemData, Result, SolveParams, solve
from pyvrp.read import ROUND_FUNCS, read
from pyvrp.stop import (
    MaxIterations,
    MaxRuntime,
    MultipleCriteria,
    NoImprovement,
)


def tabulate(headers: list[str], rows: np.ndarray) -> str:
    """
    Creates a simple table from the given header and row data.
    """
    # These lengths are used to space each column properly.
    lens = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            lens[idx] = max(lens[idx], len(str(cell)))

    header = [
        "  ".join(f"{hdr:<{ln}s}" for ln, hdr in zip(lens, headers)),
        "  ".join("-" * ln for ln in lens),
    ]

    content = [
        "  ".join(f"{c!s:>{ln}s}" for ln, c in zip(lens, r)) for r in rows
    ]

    return "\n".join(header + content)


def write_solution(where: Path, data: ProblemData, result: Result):
    with open(where, "w") as fh:
        if data.num_vehicle_types == 1:
            fh.write(str(result.best))
            fh.write(f"Cost: {round(result.cost(), 2)}\n")
            return

        # Since there are multiple vehicle types, we need to take some care
        # to assign the routes to the proper vehicle. We print all routes,
        # including empty ones. The route indices correspond to the vehicles.
        type2vehicle = [
            (int(vehicle) for vehicle in vehicle_type.name.split(","))
            for vehicle_type in data.vehicle_types()
        ]

        routes = [f"Route #{idx + 1}:" for idx in range(data.num_vehicles)]
        for route in result.best.routes():
            vehicle = next(type2vehicle[route.vehicle_type()])
            routes[vehicle] += " " + " ".join(map(str, route))

        fh.writelines(route + "\n" for route in routes)
        fh.write(f"Cost: {round(result.cost(), 2)}\n")


def _solve(
    data_loc: Path,
    round_func: str,
    seed: int,
    max_runtime: float,
    max_iterations: int,
    no_improvement: int,
    per_client: bool,
    stats_dir: Optional[Path],
    sol_dir: Optional[Path],
    **kwargs,
) -> tuple[str, str, float, int, float]:
    """
    Solves a single VRPLIB instance.

    Parameters
    ----------
    data_loc
        Filesystem location of the VRPLIB instance.
    round_func
        Rounding function to use for rounding non-integral data. Argument is
        passed to ``read()``.
    seed
        Seed to use for the RNG.
    max_runtime
        Maximum runtime (in seconds) for solving.
    max_iterations
        Maximum number of iterations for solving.
    no_improvement
        Maximum number of iterations without improvement.
    per_client
        Whether to scale stopping criteria values by the number of clients.
    stats_dir
        The directory to write runtime statistics to.
    sol_dir
        The directory to write the best found solutions to.

    Returns
    -------
    tuple[str, str, float, int, float]
        A tuple containing the instance name, whether the solution is feasible,
        the solution cost, the number of iterations, and the runtime.
    """
    if kwargs.get("config_loc"):
        params = SolveParams.from_file(kwargs["config_loc"])
    else:
        params = SolveParams()

    data = read(data_loc, round_func)

    if per_client:
        max_runtime *= data.num_clients
        max_iterations *= data.num_clients
        no_improvement *= data.num_clients

    stop = MultipleCriteria(
        [
            MaxRuntime(max_runtime),
            MaxIterations(max_iterations),
            NoImprovement(no_improvement),
        ]
    )

    result = solve(data, stop, seed, bool(stats_dir), params=params)
    instance_name = data_loc.stem

    if stats_dir:
        stats_dir.mkdir(parents=True, exist_ok=True)  # just in case
        result.stats.to_csv(stats_dir / (instance_name + ".csv"))

    if sol_dir:
        sol_dir.mkdir(parents=True, exist_ok=True)  # just in case
        write_solution(sol_dir / (instance_name + ".sol"), data, result)

    return (
        instance_name,
        "Y" if result.is_feasible() else "N",
        round(result.cost(), 2),
        result.num_iterations,
        round(result.runtime, 3),
    )


def benchmark(instances: list[Path], num_procs: int, **kwargs):
    """
    Solves a list of instances, and prints a table with the results. Any
    additional keyword arguments are passed to ``solve()``.

    Parameters
    ----------
    instances
        Paths to the VRPLIB instances to solve.
    num_procs
        Number of processors to use. Default 1.
    kwargs
        Any additional keyword arguments to pass to the solving function.
    """
    args = sorted(instances)
    func = partial(_solve, **kwargs)

    if len(instances) == 1:
        res = [func(args[0])]
    else:
        res = process_map(func, args, max_workers=num_procs, unit="instance")

    dtypes = [
        ("inst", "U37"),
        ("ok", "U1"),
        ("obj", float),
        ("iters", int),
        ("time", float),
    ]

    data = np.asarray(res, dtype=dtypes)
    headers = ["Instance", "OK", "Obj.", "Iters. (#)", "Time (s)"]

    print("\n", tabulate(headers, data), "\n", sep="")
    print(f"     Avg. objective: {data['obj'].mean():.0f}")
    print(f"    Avg. iterations: {data['iters'].mean():.0f}")
    print(f"      Avg. run-time: {data['time'].mean():.2f}s")
    print(f"       Total not OK: {np.count_nonzero(data['ok'] == 'N')}")


def main():
    description = """
    This program is a command line interface for solving VRPs, specified in
    VRPLIB format. The program can solve one or multiple such VRP instances,
    and outputs useful information in either case.
    """
    parser = argparse.ArgumentParser(prog="pyvrp", description=description)

    msg = "One or more paths to the VRPLIB instance(s) to solve."
    parser.add_argument("instances", nargs="+", type=Path, help=msg)

    msg = """
    Directory to store runtime statistics in, as CSV files (one per instance).
    """
    parser.add_argument("--stats_dir", type=Path, help=msg)

    msg = """
    Directory to store best observed solutions in, in VRPLIB format (one file
    per instance).
    """
    parser.add_argument("--sol_dir", type=Path, help=msg)

    parser.add_argument(
        "--round_func",
        default="none",
        choices=ROUND_FUNCS.keys(),
        help="Round function to apply for non-integral data. Default 'none'.",
    )

    msg = """
    Optional parameter configuration file (in TOML format). These arguments
    replace the defaults if a file is passed; default parameters are used when
    this argument is not given.
    """
    parser.add_argument("--config_loc", type=Path, help=msg)

    msg = "Seed to use for reproducible results."
    parser.add_argument("--seed", required=True, type=int, help=msg)

    msg = "Number of processors to use for solving instances. Default 1."
    parser.add_argument("--num_procs", type=int, default=1, help=msg)

    stop = parser.add_argument_group("Stopping criteria")

    msg = "Maximum runtime for each instance, in seconds."
    stop.add_argument(
        "--max_runtime", type=float, default=float("inf"), help=msg
    )

    msg = "Maximum number of iterations for solving each instance."
    stop.add_argument(
        "--max_iterations", type=int, default=float("inf"), help=msg
    )

    msg = "Maximum number of iterations without improvement."
    stop.add_argument(
        "--no_improvement", type=int, default=float("inf"), help=msg
    )

    msg = "Whether to scale stopping criteria values by the number of clients."
    stop.add_argument("--per_client", action="store_true")

    benchmark(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
