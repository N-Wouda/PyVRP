from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyvrp import Client, ProblemData, Route


def plot_route_schedule(
    data: ProblemData,
    route: Route,
    legend: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plots a route schedule. This function plots multiple time statistics
    as a function of distance travelled:

    * Solid: earliest possible trajectory / time, using time warp if the route
      is infeasible.
    * Shaded: slack up to latest possible trajectory / time, only if no time
      warp on the route.
    * Dash-dotted: driving and service time, excluding wait time and time warp.
    * Dotted: pure driving time.
    * Grey shaded background: remaining load in the vehicle.

    Parameters
    ----------
    data
        Data instance for which to plot the route schedule.
    route
        Route (list of clients) whose schedule to plot.
    legend
        Whether or not to show the legends. Default True.
    title
        Title to add to the plot.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if not ax:
        _, ax = plt.subplots()

    vehicle_type = data.vehicle_type(route.vehicle_type())
    distances = data.distance_matrix(vehicle_type.profile)
    durations = data.duration_matrix(vehicle_type.profile)
    horizon = vehicle_type.tw_late - vehicle_type.tw_early

    # Initialise tracking variables
    t = route.release_time()
    drive_time = 0
    serv_time = 0
    dist = 0
    load = route.delivery()
    slack = horizon

    # Traces and objects used for plotting
    trace_time = []
    trace_drive = []
    trace_drive_serv = []
    trace_load = []
    timewindow_lines = []
    timewarp_lines = []

    def add_traces(dist, t, drive_time, serv_time, load):
        trace_time.append((dist, t))
        trace_drive.append((dist, drive_time))
        trace_drive_serv.append((dist, drive_time + serv_time))
        trace_load.append((dist, load))

    add_traces(dist, t, drive_time, serv_time, load)
    add_traces(dist, t, drive_time, serv_time, load)

    prev_idx = vehicle_type.start_depot
    for idx in [*list(route), vehicle_type.end_depot]:
        stop = data.location(idx)

        if isinstance(stop, Client):
            tw_early = stop.tw_early
            tw_late = stop.tw_late
        else:
            tw_early = vehicle_type.tw_early
            tw_late = vehicle_type.tw_late

        delta_time = durations[prev_idx, idx]
        delta_dist = distances[prev_idx, idx]
        t += delta_time
        drive_time += delta_time
        dist += delta_dist

        add_traces(dist, t, drive_time, serv_time, load)

        if t < tw_early:
            t = tw_early

        slack = min(slack, tw_late - t)
        if t > tw_late:
            timewarp_lines.append(((dist, t), (dist, tw_late)))
            t = tw_late

        if isinstance(stop, Client):
            load -= stop.delivery
            load += stop.pickup

        add_traces(dist, t, drive_time, serv_time, load)

        if isinstance(stop, Client):
            t += stop.service_duration
            serv_time += stop.service_duration

            add_traces(dist, t, drive_time, serv_time, load)

            timewindow_lines.append(((dist, tw_early), (dist, tw_late)))

        prev_idx = idx

    # Plot primary traces
    xs, ys = zip(*trace_time)
    ax.plot(xs, ys, label="Time (earliest)")
    if slack > 0:
        ax.fill_between(
            xs, ys, np.array(ys) + slack, alpha=0.3, label="Time (slack)"
        )

    ax.plot(
        *zip(*trace_drive_serv), linestyle="-.", label="Drive + service time"
    )
    ax.plot(*zip(*trace_drive), linestyle=":", label="Drive time")

    # Plot time windows & time warps
    lc_time_windows = LineCollection(
        timewindow_lines,
        colors="grey",
        linewidths=1,
        label="Time window",
    )
    ax.add_collection(lc_time_windows)
    lc_timewarps = LineCollection(
        timewarp_lines,
        colors="red",
        linewidths=2,
        alpha=0.7,
        label="Time warp",
    )
    ax.add_collection(lc_timewarps)
    ax.set_ylim(
        bottom=vehicle_type.tw_early,
        top=max(vehicle_type.tw_late, max(t for _, t in trace_time)),
    )

    # Plot remaining load on second axis
    twin1 = ax.twinx()
    twin1.fill_between(
        *zip(*trace_load), color="black", alpha=0.1, label="Load in vehicle"
    )
    twin1.set_ylim([0, vehicle_type.capacity])

    # Set labels, legends and title
    ax.set_xlabel("Distance")
    ax.set_ylabel("Time")
    twin1.set_ylabel(f"Load (capacity = {vehicle_type.capacity:.0f})")

    if legend:
        twin1.legend(loc="upper right")
        ax.legend(loc="upper left")

    if title:
        ax.set_title(title)
