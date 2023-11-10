from typing import Dict, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fire
import h5py

from hw2d.utils.plot.movie import get_extended_viridis
from hw2d.utils.latex_format import latex_format


def is_zero_included(vals: np.ndarray, frac: float = 0.025) -> None:
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals - min_vals
    if (min_vals - frac * range_vals <= 0) and (0 <= max_vals - frac * range_vals):
        return True
    else:
        return False


def add_axes(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    ax: plt.Axes,
    linewidth: float = 1,
    linestyle: str = "solid",
    alpha: float = 1,
    color: str = "black",
) -> None:
    if is_zero_included(x_vals):
        ax.axvline(
            0, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color
        )
    if is_zero_included(y_vals):
        ax.axhline(
            0, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color
        )


def plot_timeline(values: np.ndarray, t0: float, dt: float, ax: plt.Axes, **kwargs):
    """plot values over time with proper axis adjustments

    :param values: y-values
    :type values: array-like
    :param t0: first time (only used to adjust x-axis)
    :type t0: float
    :param dt: time step (only used to adjust x-axis)
    :type dt: float
    :param ax: axis to plot on
    :type ax: matplotlib.axis
    :return: axis.plot return
    :rtype: axis.plot return
    """
    # Values
    y = values
    length = len(values)
    time = length * dt
    x = np.arange(t0, t0 + time, dt)
    x = x[: len(y)]
    # Axes
    add_axes(x, y, ax)
    # Plot
    obj = ax.plot(x, y, **kwargs)
    # Limits
    ax.set_xlim(t0, t0 + time)
    if (np.min(y) == 0) or (
        (np.min(y) > 0) and (np.min(y) - 0.01 * (np.max(y) - np.min(y)) < 0)
    ):
        ylims = ax.get_ylim()
        ylims = (0, ylims[1])
        ax.set_ylim(ylims)
    return obj


def plot_timeline_with_stds(
    y: np.ndarray,
    ax: plt.Axes,
    t0: float,
    dt: float,
    y_std: np.ndarray or None = None,
    name: str = "",
    add_label: bool = False,
    linewidth: float = 1,
    alpha: float = 0.2,
) -> Tuple[Tuple, str]:
    # Values
    length = len(y)
    time = length * dt
    x = np.arange(t0, t0 + time, dt)
    x = x[: len(y)]
    # Axes
    add_axes(x, y, ax)
    # Setup Plotting
    elements = []
    label = f"{latex_format(name)} "
    # Shadow
    if y_std is not None:
        e = ax.fill_between(x, y - y_std, y + y_std, alpha=alpha)
        elements.append(e)
        if add_label:
            label += " $\\mu \\pm \\sigma_\\mu$"
    # Timeline
    e = ax.plot(x, y, linestyle="-", linewidth=linewidth)
    elements.append(e[0])
    # Limits
    ax.set_xlim(t0, t0 + time)
    if (np.min(y) == 0) or (
        (np.min(y) > 0) and (np.min(y) - 0.01 * (np.max(y) - np.min(y)) < 0)
    ):
        ylims = ax.get_ylim()
        ylims = (0, ylims[1])
        ax.set_ylim(ylims)
    return tuple(elements), label


def plot_timetraces(
    file_path: str,
    out_path: str or None = None,
    properties: List = ("gamma_n", "gamma_c"),
    # properties: List = ("enstrophy", "energy", "kinetic_energy", "thermal_energy"),
    t0: int = 0,
    t0_std: float = 300,
):
    with h5py.File(file_path, "r") as hf:
        parameters = dict(hf.attrs)
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        t0_idx = int(t0 // parameters["dt"])
        max_len = 0
        for prop in properties:
            max_len = max(max_len, len(hf[prop]))
            if t0_idx >= len(hf[prop]):
                print(
                    f"{prop}: Not sufficient data. Range selected starts at t0={t0} to plot, data ends at t={len(hf[prop])*parameters['dt']:.2f}"
                )
                return
        t0_std_idx = int(t0_std // parameters["dt"])
        if t0_std_idx > max_len:
            print("WARNING start of statistics index is bigger than file length!")
            print(f"Calculating stats now from t0: {t0_std} -> {t0}")
            t0_std = t0
            t0_std_idx = t0_idx
        age = hf[list(hf.keys())[0]].shape[0] * parameters["dt"]
        elements = []
        labels = []
        for property in properties:
            prop_std = np.std(hf[property][t0_std_idx:])
            prop_mean = np.mean(hf[property][t0_std_idx:])
            prop_data = hf[property][t0_idx:]
            if not len(prop_data):
                print(f"WARNING no data in: {property}")
                continue
            element, label = plot_timeline_with_stds(
                prop_data,
                t0=t0,
                dt=parameters["dt"],
                ax=ax,
                name=property,
                add_label=False,
            )
            label += f" = {prop_mean:.2f}$\\pm${prop_std:.2f}"
            labels.append(label)
            elements.append(element[0])
        ax.legend(elements, labels)
        ax.set_ylabel("value")
        ax.set_xlabel("time (t)")
        ax.xaxis.set_ticks(range(0, int(age) + 1, 100))
        fig.tight_layout()
        if out_path is None:
            out_path = (
                str(file_path).split(".h5")[0] + f"_{'-'.join(properties)}" + ".jpg"
            )
        fig.savefig(out_path)
        print(out_path)


if __name__ == "__main__":
    fire.Fire(plot_timetraces)
