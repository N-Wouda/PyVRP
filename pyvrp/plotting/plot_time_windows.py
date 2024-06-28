from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyvrp import ProblemData


def plot_time_windows(
    data: ProblemData,
    title: str = "Time windows",
    ax: Optional[plt.Axes] = None,
):
    """
    Plots client time windows, as vertical bars sorted by time window.

    Parameters
    ----------
    data
        Data instance.
    title
        Title to add to the plot.
    ax
        Axes object to draw the plot on. One will be created if not provided.
    """
    if not ax:
        _, ax = plt.subplots()

    tw = np.array([[c.tw_early, c.tw_late] for c in data.clients()])

    # Lexicographic sort so for equal start we get shorter TW first
    tw = tw[np.lexsort((tw[:, 1], tw[:, 0]))]

    lines = [
        ((loc, early), (loc, late))
        for loc, (early, late) in enumerate(tw, data.num_depots - 1)
    ]
    ax.add_collection(LineCollection(lines, linewidths=1))
    ax.set_xlim([0, data.num_locations])
    ax.set_ylim([tw.min(), tw.max()])

    ax.set_title(title)
    ax.set_xlabel("Client (sorted by TW)")
    ax.set_ylabel("Time window")
