from typing import Dict, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fire
import h5py

from hw2d.utils.plot.movie import get_extended_viridis
from hw2d.utils.latex_format import latex_format


def is_zero_included(vals: np.ndarray, frac: float = 0.025) -> None:
    """Check whether zero should be included as being close to the values

    Args:
        vals (np.ndarray): Array of values
        frac (float, optional): Fraction larger or smaller that should include zero. Defaults to 0.025.

    Returns:
        bool: Whether zero should be included in the axis or not
    """
    min_yvals = np.min(vals)
    max_yvals = np.max(vals)
    range_vals = max_yvals - min_yvals
    if (min_yvals - frac * range_vals <= 0) and (0 <= max_yvals - frac * range_vals):
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
    """Mark the 0 values with lines on the x-axis and y-axis if they are close or included in the values"""
    if is_zero_included(x_vals):
        ax.axvline(
            0, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color
        )
    if is_zero_included(y_vals):
        ax.axhline(
            0, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color
        )


def get_uniform_x(values, dt, t0):
    length = len(values)
    time = length * dt
    x = np.arange(t0, t0 + time, dt)
    x = x[: length]
    return x


def plot_timeline(y: np.ndarray, x: np.ndarray, t0: float, ax: plt.Axes, **kwargs):
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
    x: np.ndarray,
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
    # Axes
    add_axes(x, y, ax)
    # Limits
    ax.set_xlim(t0, t0 + x[-1])
    return tuple(elements), label


def plot_timetraces(
    file_path: str,
    out_path: str or None = None,
    properties: List = ("gamma_c", "gamma_n", "gamma_n_spectral"),
    # properties: List = ("energy", "kinetic_energy", "thermal_energy", "enstrophy", "enstrophy_phi"),
    t0: int = 0,
    t0_std: float = 300,
    xtick_interval: int = 100,
    y_cutoff: int = 1e5,
):
    with h5py.File(file_path, "r") as hf:
        # Loading data
        parameters = dict(hf.attrs)
        adaptive_step_size = parameters.get("adaptive_step_size", False)
        # Time handling
        init_time = parameters.get("initial_time", 0)
        if adaptive_step_size:
            times_x = hf["time"][:, 0]
            # Use argmin on the absolute difference between the array elements and the target value
            t0_idx = np.argmin(np.abs(times_x - t0))
            age = times_x[-1]
        else:  # Fixed step size
            t0_idx = int(t0 // parameters["frame_dt"])
            age = (hf[list(hf.keys())[0]].shape[0] * parameters["frame_dt"]) + init_time
        # Determine max length of properties
        max_len = 0
        for prop in properties:
            max_len = max(max_len, len(hf[prop]))
            # Check for sufficient data for the requested interval
            if t0_idx >= len(hf[prop]):
                print(
                    f"{prop}: Not sufficient data. Range selected starts at t0={t0} to plot, data ends at t={len(hf[prop])*parameters['dt']:.2f}"
                )
                return
        if max_len < t0_idx:
            print(f"Note: Max length of properties {max_len:,} <  {t0_idx:,} Requested start time")
            return
        # Time handling for standard deviation calculations
        if adaptive_step_size:
            t0_std_idx = np.argmin(np.abs(times_x - (t0_std - init_time)))
        else:
            t0_std_idx = int((t0_std - init_time) // parameters["frame_dt"])
        if t0_std_idx > max_len:
            print("WARNING start of statistics index is bigger than file length!")
            print(f"Calculating stats now from t0: {t0_std:,} -> {t0:,}")
            t0_std = t0
            t0_std_idx = t0_idx
        # Plot elements
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        elements = []
        labels = []
        min_yval = 0
        max_yval = 0
        for prop in properties:
            # Calculate statistical properties
            prop_std = np.std(hf[prop][t0_std_idx:])
            prop_mean = np.mean(hf[prop][t0_std_idx:])
            # Properties for plotting
            prop_data = hf[prop][t0_idx:]
            if not len(prop_data):
                print(f"WARNING no data in: {prop}")
                continue
            # Generate x-values
            if adaptive_step_size:
                x = times_x[t0_idx:]
            else:
                time = len(prop_data) * parameters["frame_dt"]
                x = np.arange(t0, t0 + time, parameters["frame_dt"])
            x = x[:len(prop_data)]  # Ensure x,y align
            # Plot elements
            element, label = plot_timeline_with_stds(
                y=prop_data,
                x=x,
                t0=t0,
                dt=parameters["frame_dt"],
                ax=ax,
                name=property,
                add_label=False,
            )
            # Axis handling
            min_yval = min(min_yval, np.min(prop_data))
            max_yval = max(max_yval, np.min(prop_data))
            label += f" = {prop_mean:.2f}$\\pm${prop_std:.2f}"
            labels.append(label)
            elements.append(element[0])
        # Adjust y-axis
        if np.isclose(min_yval, 0, rtol=1e-05, atol=1e-08, equal_nan=False) or (
            (min_yval > 0) and (min_yval - 0.01 * (max_yval - min_yval) < 0)
        ):
            ylims = ax.get_ylim()
            ylims = (0, ylims[1])
            ax.set_ylim(ylims)
        if max_yval > y_cutoff:
            print(f"WARNING diverged. Setting limit to: {y_cutoff}")
            ax.set_ylim((0, y_cutoff))
        # Adjust Labels
        ax.legend(elements, labels)
        ax.set_ylabel("value")
        ax.set_xlabel("time (t)")
        if age // xtick_interval < 15 and age // xtick_interval > 2:
            ax.xaxis.set_ticks(range(t0, int(age) + 1, xtick_interval))
        # Wrap up figure
        fig.tight_layout()
        # Save
        if out_path is None:
            out_path = (
                str(file_path).split(".h5")[0] + f"_{'-'.join(properties)}" + ".jpg"
            )
        fig.savefig(out_path)
        print(f"saved to:  {out_path}")


def process_property(hf, property_name, parameters, ax, t0, t0_idx, t0_std_idx, min_yval, max_yval, set_label=None):
    # Calculate statistical properties
    prop_std = np.std(hf[property_name][t0_std_idx:])
    prop_mean = np.mean(hf[property_name][t0_std_idx:])
    # Properties for plotting
    parameters = dict(hf.attrs)
    adaptive_step_size = parameters.get("adaptive_step_size", False)
    prop_data = hf[property_name][t0_idx:]
    if not len(prop_data):
        print(f"WARNING no data in: {property_name}")
        return None, None
    # X-Axis values
    if adaptive_step_size:
        times_x = hf["time"][:, 0]
        x = times_x[t0_idx:]
        end_time = times_x[-1]
    else:
        end_time = len(prop_data) * parameters["frame_dt"]
        x = np.arange(t0, t0 + end_time, parameters["frame_dt"])
    x = x[:len(prop_data)]  # Ensure x,y align
    # Plot elements
    element, label = plot_timeline_with_stds(
        y=prop_data,
        x=x,
        t0=t0,
        dt=parameters["frame_dt"],
        ax=ax,
        name=property_name,
        add_label=False,
    )
    if set_label:
        label = set_label
    # Axis handling
    min_yval = min(min_yval, np.min(prop_data))
    max_yval = max(max_yval, np.min(prop_data))
    # Format label
    label += f" = {prop_mean:.2f}$\\pm${prop_std:.2f}"
    return element, label


def format_axis(ax, min_yval, max_yval, y_cutoff, elements, labels, t0, age, xtick_interval, ylabel, xlabel):
    # Adjust y-axis
    if np.isclose(min_yval, 0, rtol=1e-05, atol=1e-08, equal_nan=False) or (
        (min_yval > 0) and (min_yval - 0.01 * (max_yval - min_yval) < 0)
    ):
        ylims = ax.get_ylim()
        ylims = (0, ylims[1])
        ax.set_ylim(ylims)
    if max_yval > y_cutoff:
        print(f"WARNING diverged. Setting limit to: {y_cutoff}")
        ax.set_ylim((0, y_cutoff))
    # Adjust Labels
    ax.legend(elements, labels)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if age // xtick_interval < 15:
        ax.xaxis.set_ticks(range(t0, int(age) + 1, xtick_interval))


def plot_timetrace_comparison(
    file_path: str,
    out_path: str or None = None,
    properties: List = ("gamma_c", "gamma_n", "gamma_n_spectral"),
    prefix: str = "fullres_",
    # properties: List = ("energy", "kinetic_energy", "thermal_energy", "enstrophy", "enstrophy_phi"),
    t0: int = 0,
    t0_std: float = 300,
    xtick_interval: int = 100,
    y_cutoff: int = 1e5,
):
    with h5py.File(file_path, "r") as hf:
        print(f"Keys: {hf.keys()}")
        parameters = dict(hf.attrs)
        fig, axarr = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)
        adaptive_step_size = parameters.get("adaptive_step_size", False)
        # Time handling
        init_time = parameters.get("initial_time", 0)
        t0 = max(t0, init_time)
        if adaptive_step_size:
            times_x = hf["time"][:, 0]
            # Use argmin on the absolute difference between the array elements and the target value
            t0_idx = np.argmin(np.abs(times_x - t0))
            end_time = times_x[-1]
            print(f"Using adaptive step size adjustment:  t0_idx={t0_idx:,.0f} ({times_x[t0_idx]:,.0f}) | x ranging from: ({times_x[0]}-{times_x[-1]})")
        else:
            t0_idx = int(t0 // parameters["frame_dt"])
            end_time = len(hf["density"]) * parameters["frame_dt"]
        # Determine max length of properties
        max_len = 0
        for prop in properties:
            max_len = max(max_len, len(hf[prop]))
            if t0_idx >= len(hf[prop]):
                print(
                    f"{prop}: Not sufficient data. Range selected starts at t0={t0} to plot, data ends at t={len(hf[prop])*parameters['frame_dt']:.2f}"
                )
                return
        # Time handling for standard deviation calculations
        if adaptive_step_size:
            t0_std_idx = np.argmin(np.abs(times_x - (t0_std - init_time)))
            print(f"Using adaptive step size adjustment:  t0_std={t0_std:,.0f} | init_time={init_time:,.4f} | t0_std_idx={t0_std_idx:,.0f}")
        else:
            t0_std_idx = int((t0_std - init_time) // parameters["frame_dt"])
        if t0_std_idx > max_len:
            print("WARNING start of statistics index is bigger than file length!")
            print(f"Calculating stats now from t0: {t0_std:,} -> {t0:,}")
            t0_std = t0
            t0_std_idx = t0_idx
        # Plot elements
        elements = []
        labels = []
        fullres_elements = []
        fullres_labels = []
        min_yval = 0
        max_yval = 0
        for property_name in properties:
            element, label = process_property(hf, property_name, parameters, axarr[0], t0, t0_idx, t0_std_idx, min_yval, max_yval)
            if element is not None and label is not None:
                labels.append(label)
                elements.append(element[0])
            # Fullres
            fullres_name = f"{prefix}{property_name}"
            if fullres_name in hf:
                fullres_element, fullres_label = process_property(hf, fullres_name, parameters, axarr[1], t0, t0_idx, t0_std_idx, min_yval, max_yval, set_label=f"Full {latex_format(property_name)} ")
                if fullres_element is not None and fullres_label is not None:
                    fullres_labels.append(fullres_label)
                    fullres_elements.append(fullres_element[0])
        format_axis(axarr[0], min_yval, max_yval, y_cutoff, elements, labels, t0, end_time, xtick_interval, ylabel="Reduced resultion", xlabel="")
        format_axis(axarr[1], min_yval, max_yval, y_cutoff, fullres_elements, fullres_labels, t0, end_time, xtick_interval, ylabel="Full resolution", xlabel="time (t)")
        # Wrap up figure
        fig.tight_layout()
        # Save
        if out_path is None:
            out_path = (
                str(file_path).split(".h5")[0] + f"_{'-'.join(properties)}" + ".jpg"
            )
        fig.savefig(out_path)
        print(f"saved to:  {out_path}")


def main(
    file_path: str,
    out_path: str or None = None,
    t0: int = 0,
    t0_std: float = 300,
    comparison=True,
):
    if comparison:
        plot_timetrace_comparison(
            file_path=file_path,
            out_path=out_path,
            properties=(
                "gamma_c",
                "gamma_n",
                "gamma_n_spectral",
                "energy",
                "kinetic_energy",
                "thermal_energy",
                "enstrophy",
                "enstrophy_phi"
            ),
            t0=t0,
            t0_std=t0_std,
        )
    else:
        plot_timetraces(
            file_path=file_path,
            out_path=out_path,
            properties=("gamma_c", "gamma_n", "gamma_n_spectral"),
            t0=t0,
            t0_std=t0_std,
        )
        plot_timetraces(
            file_path=file_path,
            out_path=out_path,
            properties=("energy", "kinetic_energy", "thermal_energy", "enstrophy_phi"), #"enstrophy",
            t0=t0,
            t0_std=t0_std,
        )


if __name__ == "__main__":
    fire.Fire(main)
