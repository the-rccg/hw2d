import math
from typing import Any, Callable, Dict, Iterable, Tuple

import fire
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


def get_extended_viridis(vals: int = 600) -> ListedColormap:
    """
    Generate an extended viridis colormap.

    Args:
        vals (int): Number of color values. Default is 600.

    Returns:
        ListedColormap: A colormap object.
    """
    VIRIDIS_EXTENDED = [
        [0.0, 255, 200, 100],
        [0.13, 255, 153, 51],
        [0.25, 230, 5, 40],
        [0.38, 150, 3, 62],
        [0.5, 68, 1, 84],
        [0.55, 72, 33, 115],
        [0.59, 67, 62, 133],
        [0.64, 56, 88, 140],
        [0.68, 45, 112, 142],
        [0.73, 37, 133, 142],
        [0.77, 30, 155, 138],
        [0.82, 42, 176, 127],
        [0.86, 82, 197, 105],
        [0.90, 34, 213, 73],
        [0.95, 194, 223, 35],
        [1.0, 253, 231, 37],
    ]
    VE = np.array(VIRIDIS_EXTENDED)
    VE[:, 1:] = VE[:, 1:] / 256
    ve_list = []
    for i in range(1, len(VIRIDIS_EXTENDED)):
        pts = int((VE[i, 0] - VE[i - 1, 0]) * vals)
        ve_list.append(
            np.stack(
                [
                    np.linspace(VE[i - 1, 1], VE[i, 1], pts, endpoint=False),
                    np.linspace(VE[i - 1, 2], VE[i, 2], pts, endpoint=False),
                    np.linspace(VE[i - 1, 3], VE[i, 3], pts, endpoint=False),
                ],
                axis=-1,
            )
        )
    VE = np.vstack(ve_list)
    return ListedColormap(VE)


def new_cbar_max(nm: float, pm: float) -> float:
    """
    Adjust maximum of colorbar using sqrt diff scaling.

    Args:
        nm (float): Negative maximum.
        pm (float): Positive maximum.

    Returns:
        float: Adjusted maximum.
    """
    if pm > nm:
        nm = nm + np.sqrt(pm - nm)
    return nm


def new_cbar_max_smooth(nm: float, pm: float) -> float:
    """
    Adjust maximum of colorbar using average scaling for a smoother effect.

    Args:
        nm (float): Negative maximum.
        pm (float): Positive maximum.

    Returns:
        float: Adjusted maximum.
    """
    if pm > nm:
        nm = (nm + pm) / 2
    return nm


def setup_visualization(
    axarr: Iterable[Any],
    plasma_steps: Iterable[Dict[str, float]],
    params: Dict[str, float],
    plot_order: Iterable[str],
    cmap: Any,
) -> Tuple[Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any]]:
    """Setup initial chart objects."""
    ims, cxs, cbs, txs = [], [], [], []
    for i, ax in enumerate(axarr):
        ax.xaxis.set_tick_params(
            which="both", bottom=False, top=False, labelbottom=False
        )
        ax.yaxis.set_tick_params(
            which="both", bottom=False, top=False, labelbottom=False
        )
        cxs.append(make_axes_locatable(ax).append_axes("right", "5%", "1%"))
        ims.append(ax.imshow(plasma_steps[plot_order[i]][0], origin="lower", cmap=cmap))
        cbs.append(plt.colorbar(ims[i], cax=cxs[i]))
        txs.append(ax.set_title(f"{plot_order[i]}"))
    return ims, cxs, cbs, txs


def setup_figure(title: str) -> Tuple[plt.Figure, np.ndarray]:
    """
    Setup the figure for visualization.

    Args:
        title (str): Title for the figure.

    Returns:
        Tuple containing the figure and array of axes.
    """
    fig, axarr = plt.subplots(1, 3, figsize=(12, 4.35), sharex=True, sharey=True)
    fig.suptitle(title)
    return fig, axarr


def time_to_length(time: int, dt: float) -> int:
    """Convert time to length based on the given time step."""
    return int(round(time / dt))


def generate_title(params: Dict[str, float]) -> str:
    return ",  ".join(
        [
            f"c1={params['c1']}",
            # f"L={params['L']:.2f}",
            f"k0={params['k0']:.2f}",
            f"pts={params['grid_pts']:.0f}",
            f"dt={params['dt']}",
            f"N={params['N']:.0f}",
            f"nu={params['nu']:.2g}",
        ]
    )


def create_movie_equally_spaced(
    input_filename: str,
    output_filename: str,
    t0: int = 0,
    t1: int = None,
    zero_omega: bool = False,
    plot_order: Iterable[str] = ["density", "omega", "phi"],
    cmap: ListedColormap = get_extended_viridis(),
    min_fps: float = 10,
    dpi: int = 75,
    speed: float = 1,
    writer: str = "ffmpeg",
) -> None:
    # Setup details
    hf = h5py.File(input_filename, "r")
    params = dict(hf.attrs)
    title = generate_title(params)
    # Time handling
    if t1 is None:
        t1 = (len(hf[plot_order[0]]) - 1) * params["frame_dt"]
    t0_idx = time_to_length(t0 - params.get("initial_time", 0), params["frame_dt"])
    t1_idx = time_to_length(t1 - params.get("initial_time", 0), params["frame_dt"])
    # Determine fps and step size to use
    fps = int(speed / params["frame_dt"])
    print(f"{speed}t/s  @  {params['frame_dt']} t/frame  implies {fps} frames/s")
    frame_steps = max(1, fps // min_fps)
    fps /= frame_steps
    print(
        f"with min_fps {min_fps} we take every {frame_steps} frames still generate {fps} frame/s"
    )
    # Define Progress
    total = int((t1_idx - t0_idx) // frame_steps)
    frame_range = range(t0_idx, t1_idx + 1, frame_steps)
    pbar = tqdm(total=total, smoothing=0.3)
    # Setup Figure & Animation
    fig, axarr = setup_figure(title)
    ims, cxs, cbs, txs = setup_visualization(
        axarr, hf, params, plot_order, cmap
    )
    fig.subplots_adjust(top=0.94, bottom=0, right=0.95, left=0.01, hspace=0, wspace=0.2)
    # Animation
    writer = animation.writers[writer](fps=fps, metadata=dict(artist="Robin Greif"))

    def animate(t_idx: int):
        """Update data for animation."""
        for i, ax in enumerate(axarr):
            field_name = plot_order[i]
            arr = hf[field_name][t_idx]
            if zero_omega and field_name == "omega":
                arr -= np.mean(arr)
            vmax = np.max(arr)
            vmin = np.min(arr)
            pmin, pmax = ims[i].get_clim()
            pm = np.max([np.abs(pmin), pmax])
            nm = np.max([np.abs(vmin), vmax])
            nm = new_cbar_max_smooth(nm, pm)
            ims[i].set_data(arr)
            ims[i].set_clim(-nm, nm)
            txs[i].set_text(f"{field_name} (t={t_idx*params['frame_dt']:.0f})")
        pbar.update(1)

    ani = animation.FuncAnimation(fig, animate, frames=frame_range)
    # Save Movie
    output_filename = f"{output_filename}_dpi={dpi}_fps={fps:.0f}_speed={speed:.0f}_t0={t0}_t1={t1}.mp4"
    ani.save(output_filename, writer=writer, dpi=dpi)
    # Close and wrap-up
    plt.close()
    hf.close()
    print(f"saved as:  {output_filename}")


def create_movie_unequal_spaced(
    input_filename: str,
    output_filename: str,
    t0: float = 0,
    t1: float = None,
    zero_omega: bool = False,
    plot_order: Iterable[str] = ["density", "omega", "phi"],
    cmap: ListedColormap = get_extended_viridis(),
    min_fps: float = 10,
    dpi: int = 75,
    speed: float = 10,
    writer: str = "ffmpeg",
    debug: bool = False,
) -> None:
    """
    Generate movie for unequally spaced timeseries. Adjusts sampling for each second. 
    NOTE: Currently has slowdowns at the second transitions, so higher fps is highly encouraged.
    To resolve this, another sampling algorithm is needed or a variable frame rate movie format to slightly adjust it.
    """
    # Setup details
    hf = h5py.File(input_filename, "r")
    params = dict(hf.attrs)
    title = generate_title(params)
    print(f"Adaptive Timestep detected! Using FPS = {min_fps}")

    # Time handling
    times = np.array(hf["time"][:,0])

    if t1 is None:
        t1 = times[-1]

    t0_idx = np.searchsorted(times, t0)
    t1_idx = np.searchsorted(times, t1)

    # Determine fps and coarse sampling
    t_per_sec = speed
    sim_duration_t = times[t1_idx] - times[t0_idx]
    movie_duration_s = int(sim_duration_t / t_per_sec)
    frame_count = t1_idx - t0_idx
    implied_fps = movie_duration_s / frame_count
    frames_per_t = math.ceil(t_per_sec / min_fps)
    #print(f"{sim_duration_t=} | {movie_duration_s=} | {frame_count=} | {implied_fps=} | {frames_per_t=}")

    # Frame selection
    selected_frames = []
    prev_t = t0
    prev_t_idx = t0_idx
    # Adjust sampling every second
    for s_i in range(1, movie_duration_s+1):
        t_i = s_i * t_per_sec
        t_i_idx = np.searchsorted(times, t_i)
        frame_count_for_s = t_i_idx - prev_t_idx
        # sample frequency
        step_size = int(round(frame_count_for_s / min_fps))
        selected_frames += list(np.arange(prev_t_idx, t_i_idx, step_size))
        if debug:
            print(f"Period: {(s_i-1) * t_per_sec}-{t_i} ({times[prev_t_idx]}-{times[t_i_idx]}) | prev_t_idx={prev_t_idx} | t_i_idx={t_i_idx} | frame_count_for_s={frame_count_for_s} | step_size={step_size} | {len(selected_frames)}")
        prev_t = t_i
        prev_t_idx = t_i_idx

    print(np.diff(np.array(selected_frames)))
    #print([my_list[i+1] - my_list[i] for i in range(len(my_list) - 1)])

    # Define Progress
    pbar = tqdm(total=len(selected_frames), smoothing=0.3)

    # Setup Figure & Animation
    fig, axarr = setup_figure(title)
    ims, cxs, cbs, txs = setup_visualization(
        axarr, hf, params, plot_order, cmap
    )
    fig.subplots_adjust(top=0.94, bottom=0, right=0.95, left=0.01, hspace=0, wspace=0.2)

    # Animation
    def animate(frame_idx: int):
        """Update data for animation."""
        t_idx = selected_frames[frame_idx]
        for i, ax in enumerate(axarr):
            field_name = plot_order[i]
            arr = hf[field_name][t_idx]
            if zero_omega and field_name == "omega":
                arr -= np.mean(arr)
            vmax = np.max(arr)
            vmin = np.min(arr)
            pmin, pmax = ims[i].get_clim()
            pm = np.max([np.abs(pmin), pmax])
            nm = np.max([np.abs(vmin), vmax])
            nm = new_cbar_max_smooth(nm, pm)
            ims[i].set_data(arr)
            ims[i].set_clim(-nm, nm)
            txs[i].set_text(f"{field_name} (t={times[t_idx]:.2f})")
        pbar.update(1)

    ani = animation.FuncAnimation(fig, animate, frames=len(selected_frames))
    
    # Save Movie
    output_filename = f"{output_filename}_dpi={dpi}_fps={min_fps:.0f}_speed={speed:.0f}_t0={t0}_t1={t1}.mp4"
    writer = animation.writers[writer](fps=min_fps, metadata=dict(artist="Robin Greif"))
    ani.save(output_filename, writer=writer, dpi=dpi)
    
    # Close and wrap-up
    plt.close()
    hf.close()
    print(f"saved as:  {output_filename}")


def create_movie(
    **kwargs
):
    # List all available writers
    available_writers = animation.writers.list()
    writer = "ffmpeg"
    if writer in available_writers:
        print(f"Writer ({writer}) not in available writers.")
        print(f"Available writers:  {available_writers}")

    hf = h5py.File(kwargs["input_filename"], "r")
    params = dict(hf.attrs)
    if params["adaptive_step_size"]:
        create_movie_unequal_spaced(**kwargs, writer=writer)
    else:
        create_movie_equally_spaced(**kwargs, writer=writer)



def main(
    input_path: str,
    output_path: str,
    t0: int = 0,
    t1: int or None = None,
    plot_order: Iterable = ("density", "omega", "phi"),
    dpi: int = 75,
    min_fps: int = 10,
    speed: int = 10,
) -> None:
    create_movie(
        input_filename=input_path,
        output_filename=output_path,
        t0=t0,
        t1=t1,
        plot_order=plot_order,
        min_fps=min_fps,
        dpi=dpi,
        speed=speed,
    )


if __name__ == "__main__":
    fire.Fire(main)
