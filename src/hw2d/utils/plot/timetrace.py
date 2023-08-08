from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import latexplotlib as lpl
from tqdm import tqdm
import fire
import h5py

from hw2d.utils.plot.movie import get_extended_viridis
from hw2d.utils.latex_format import latex_format


def plot_dict(
    plot_dic,
    cmap,
    couple_cbars=False,
    figsize=None,
    sharex=False,
    sharey=False,
    vertical=False,
    equal=True,
    cbar_label_spacing=None,
):
    n = len(plot_dic.values())
    labels = list(plot_dic.keys())
    if figsize is None:
        figsize = (n * 4, 4)
    if vertical:
        fig, axarr = lpl.subplots(n, 1, figsize=figsize, sharex=sharex)
    else:
        fig, axarr = lpl.subplots(
            1, n, figsize=figsize, ratio=1, constrained_layout=True
        )
    vmin = 0
    vmax = 0
    imgs = []
    cbars = []
    for i in tqdm(range(n)):
        # Local Names
        label = labels[i]
        data = plot_dic[label]
        ax = axarr[i]
        if len(data.shape) == 1:
            ax.axhline(0, color="k", alpha=0.7, linewidth=1)
            ax.plot(data)
        elif len(data.shape) == 2:
            imgs.append(ax.imshow(data, cmap=cmap))  # , interpolation='nearest'))
            cbars.append(plt.colorbar(imgs[-1], pad=0, fraction=0.05))  # , cax=cax
            cax = cbars[-1].ax
            if cbar_label_spacing is not None:
                cax.yaxis.set_tick_params(pad=cbar_label_spacing)
            cax.xaxis.set_tick_params(pad=0)
            # cbars.append(plt.colorbar(imgs[-1], cax=cax))
        else:
            raise BaseException(
                f"Shape not recognized.  {label} has shape {data.shape}"
            )
        # Properties
        ax.set_title(label, pad=4)
        vmin = min(np.min(data), vmin)
        vmax = max(np.max(data), vmax)
        if equal:
            ax.set_aspect("equal")
    # fix colorbars together
    if couple_cbars:
        for img in imgs:
            img.set_clim(vmin, vmax)
    return fig


def is_zero_included(vals, frac: float = 0.025) -> None:
    min_vals = np.min(vals)
    max_vals = np.max(vals)
    range_vals = max_vals - min_vals
    if (min_vals - frac * range_vals <= 0) and (0 <= max_vals - frac * range_vals):
        return True
    else:
        return False


def add_axes(
    x_vals,
    y_vals,
    ax,
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


def plot_timeline(values, t0: float, dt: float, ax, **kwargs):
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
    y,
    ax,
    t0: float,
    dt: float,
    y_std=None,
    name: str = "",
    add_label: bool = False,
    linewidth: float = 1,
    alpha: float = 0.2,
    **kwargs,
) -> Tuple[Tuple, str]:
    # Values
    length = len(y)
    time = length * dt
    x = np.arange(t0, t0 + time, dt)
    x = x[: len(y)]
    # Axes
    add_axes(x, y, ax)
    # Limits
    ax.set_xlim(t0, t0 + time)
    if (np.min(y) == 0) or (
        (np.min(y) > 0) and (np.min(y) - 0.01 * (np.max(y) - np.min(y)) < 0)
    ):
        ylims = ax.get_ylim()
        ylims = (0, ylims[1])
        ax.set_ylim(ylims)
    # Setup Plotting
    elements = []
    label = f"{latex_format(name)} "
    # Shadow
    if y_std is not None:
        e = ax.fill_between(x, y - y_std, y + y_std, alpha=alpha)
        elements.append(e)
        if add_label:
            label += " $\mu \pm \sigma_\mu$"
    # Timeline
    e = ax.plot(
        x, y, linestyle="-", linewidth=linewidth
    )  # , label=f"{name}: mean of means")
    elements.append(e[0])
    return tuple(elements), label


def main(
    file_path: str = "/ptmp/rccg/hw2d/test.h5",  # "/ptmp/rccg/hw2d/512x512_dt=0.025_nu=1.0e-09.h5",
    properties: List = ("gamma_n", "gamma_c"),
    t0: int = 0,
    t0_std: float = 300,
):
    with h5py.File(file_path, "r") as hf:
        print(hf)
        parameters = dict(hf.attrs)
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        t0_std_idx = int(t0_std // parameters["dt"])
        age = hf[list(hf.keys())[0]].shape[0] * parameters["dt"]
        print(age, t0_std_idx)
        elements = []
        labels = []
        for property in properties:
            prop_std = np.std(hf[property][t0_std_idx:])
            prop_mean = np.mean(hf[property][t0_std_idx:])
            # plot_timeline(hf[property][:], t0=t0, dt=parameters["dt"], ax=ax)
            element, label = plot_timeline_with_stds(
                hf[property][:],
                t0=t0,
                dt=parameters["dt"],
                ax=ax,
                name=property,
                add_label=False,
            )
            label += f" = {prop_mean:.3f}$\pm${prop_std:.3f}"
            labels.append(label)
            elements.append(element[0])
        ax.legend(elements, labels)
        ax.set_ylabel("value")
        ax.set_xlabel("time (t)")
        ax.xaxis.set_ticks(range(0, int(age) + 1, 100))
        fig.tight_layout()
        fig.savefig("test.jpg")
        print("test.jpg")


if __name__ == "__main__":
    fire.Fire(main)
