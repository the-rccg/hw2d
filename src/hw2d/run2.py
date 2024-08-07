import fire
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from typing import Iterable

# Local imports
from hw2d.model import HW
from hw2d.initializations.fourier_noise import get_fft_noise
from hw2d.initializations.sine import get_2d_sine
from hw2d.utils.io import (
    get_save_params,
    create_appendable_h5,
    save_to_buffered_h5,
    append_h5,
    continue_h5_file,
)
from hw2d.utils.namespaces import Namespace
from hw2d.utils.plot.movie import create_movie
from hw2d.utils.run_properties import calculate_properties
from hw2d.utils.plot.timetrace import plot_timetraces, plot_timetrace_comparison
from hw2d.utils.downsample import fourier_downsample
from hw2d.utils.helpers import format_print_dict
from hw2d.physical_properties.numpy_properties import (
    get_gamma_n,
    get_gamma_n_spectrally,
    get_gamma_c,
    get_energy,
    get_energy_N_spectrally,
    get_energy_V_spectrally,
    get_enstrophy,
    get_enstrophy_phi,
    get_DE,
    get_DU,
    get_dE_dt,
    get_dU_dt,
)


property_fncs = {
    "gamma_n": lambda n, p, dx, **kwargs: get_gamma_n(n=n, p=p, dx=dx),
    "gamma_n_spectral": lambda n, p, dx, **kwargs: get_gamma_n_spectrally(
        n=n, p=p, dx=dx
    ),
    "gamma_c": lambda n, p, dx, c1, **kwargs: get_gamma_c(n=n, p=p, c1=c1, dx=dx),
    "energy": lambda n, p, dx, **kwargs: get_energy(n=n, phi=p, dx=dx),
    "thermal_energy": lambda n, **kwargs: get_energy_N_spectrally(n=n),
    "kinetic_energy": lambda p, dx, **kwargs: get_energy_V_spectrally(p=p, dx=dx),
    "enstrophy": lambda n, o, dx, **kwargs: get_enstrophy(n=n, omega=o, dx=dx),
    "enstrophy_phi": lambda n, p, dx, **kwargs: get_enstrophy_phi(n=n, phi=p, dx=dx),
    "DE": lambda n, p, Dn, Dp, **kwargs: get_DE(n=n, p=p, Dn=Dn, Dp=Dp),
    "DU": lambda n, o, Dn, Dp, **kwargs: get_DU(n=n, o=o, Dn=Dn, Dp=Dp),
    "dE_dt": lambda gamma_n, gamma_c, DE, **kwargs: get_dE_dt(gamma_n=gamma_n, gamma_c=gamma_c, DE=DE),
    "dU_dt": lambda gamma_n, DU, **kwargs: get_dU_dt(gamma_n=gamma_n, DU=DU),
}


def run(
    # Physics & Numerics
    step_size: float = 0.025,
    adaptive_step_size: bool = False,
    end_time: float = 1_000,
    grid_pts: int = 512,
    k0: float = 0.15,
    N: int = 3,
    nu: float = 5.0e-09,
    c1: float = 1,
    kappa_coeff: float = 1.0,
    poisson_bracket_coeff: float = 1.0,
    zonal: bool = False,
    # Running
    show_property: str = "gamma_n",
    # Initialization
    seed: int or None = None,
    init_type: str = "normal",
    init_scale: float = 1 / 100,
    # Saving
    output_path: str = "c1=1.0_nu=5e-09_dt=0.025_v=1.h5",
    recording_start_time: float = 0,
    continue_file: bool = False,
    buffer_length: int = 100,
    snaps: int = 1,
    chunk_size: int = 100,
    downsample_factor: float = 16,
    add_last_state: bool = True,
    # Movie
    movie: bool = False,
    min_fps: int = 10,
    dpi: int = 75,
    speed: int = 10,
    # Properties
    properties: Iterable[str] = [
        "gamma_n",
        "gamma_n_spectral",
        "gamma_c",
        "energy",
        "thermal_energy",
        "kinetic_energy",
        "enstrophy",
        "enstrophy_phi",
    ],
    # Plotting
    t0_std=500,
    plot_properties: Iterable[str] = (
        "gamma_c",
        "gamma_n",
        "gamma_n_spectral",
        "energy",
        "kinetic_energy",
        "thermal_energy",
        "enstrophy",
        "enstrophy_phi"
    ),
    # Other
    debug: bool = False,
    force_recompute: bool = False,
):
    """
    Run the simulation with the given parameters.

    Args:
        step_size (float, optional): Incremental step for simulation progression. Defaults to 0.025.
        end_time (float, optional): Duration till the simulation should run. Defaults to 1_000.
        grid_pts (int, optional): Grid points. Suggested: 128 for coarse, 1024 for fine. Defaults to 512.
        k0 (float, optional): Determines k-focus. Suggested: 0.15 for high-k, 0.0375 for low-k. Defaults to 0.15.
        N (int, optional): Dissipation exponent's half value. Defaults to 3.
        nu (float, optional): Viscosity. Suggested: 5e-10 for coarse-large, 1e-4 for fine-small. Defaults to 5.0e-08.
        c1 (float, optional): Transition scale between hydrodynamic and adiabatic. Suggested values: 0.1, 1, 5. Defaults to 1.0.
        kappa_coeff (float, optional): Coefficient of d/dy phi. Defaults to 1.0.
        poisson_bracket_coeff (float, optional): Coefficient of Poisson bracket [A,B] implemented with Arakawa Scheme. Defaults to 1.0.
        zonal (bool, ooptional): If True, uses make_zonal_func to simulate mHW2D. Defaults to False.
        seed (int or None, optional): Seed for random number generation. Defaults to None.
        init_type (str, optional): Initialization method. Choices: 'fourier', 'sine', 'random', 'normal'. Defaults to 'normal'.
        init_scale (float, optional): Scaling factor for initialization. Defaults to 0.01.
        output_path (str, optional): Where to save the simulation data. Defaults to ''.
        recording_start_time (float, optional): Time (t) from which onwards the data is recorded. Defaults to 0.
        continue_file (bool, optional): If True, continue with existing file. Defaults to False.
        buffer_length (int, optional): Size of buffer for storage. Defaults to 100.
        chunks (int, optional): Chunks of the h5 file. Defaults to 100.
        downsample_factor (float, optional): Factor along each axis that it is downsampled with for saving. Defaults to 1.
        add_last_state (bool, optional): Whether the last high-resolution frame is saved. Turns True if downsampling on. Defaults to False.
        snaps (int, optional): Snapshot intervals for saving. Defaults to 1.
        movie (bool, optional): If True, generate a movie out of simulation. Defaults to True.
        min_fps (int, optional): Parameter for movie generation. Defaults to 10.
        dpi (int, optional): Parameter for movie generation. Defaults to 75.
        speed (int, optional): Parameter for movie generation. Defaults to 5.
        properties (Iterable[str], optional): List of properties to calculate for the saved file.
        plot_properties (Iterable[str], optional): List of properties to plot a timetrace for.
        debug (bool, optional): Enable or disable debug mode. Defaults to False.

    Returns:
        None: The function saves simulation data or generates a movie as specified.
    """
    step_size = float(step_size)
    nu = float(nu)

    # Unpacking
    y = grid_pts
    x = grid_pts
    L = 2 * np.pi / k0  # Physical box size
    dx = L / x  # Grid resolution
    steps = int(end_time / step_size)  # Number of Steps until end_time
    snap_count = steps // snaps + 1  # number of snapshots
    field_list = ("density", "omega", "phi")
    initial_time = np.float64(0)
    iteration_count = 0
    np.random.seed(seed)
    if downsample_factor != 1:
        add_last_state = True
    if show_property not in properties:
        properties += [show_property]

    # Define Initializations
    noise = {
        "fourier": lambda y, x: get_fft_noise(
            resolution=[y, x],
            size=L,
            scale=1,
            min_wavelength=dx * 12,
            max_wavelength=dx * grid_pts * 100,
            factor=2,
        ),
        "sine": lambda y, x: get_2d_sine((y, x), L),
        "random": lambda y, x: np.random.rand(y, x).astype(np.float64),
        "normal": lambda y, x: np.random.normal(size=(y, x)).astype(np.float64),
    }
    downsample_fnc = fourier_downsample

    # Physics
    # TODO: Move all to Tensors along batch dimension for simulations!
    physics_params = dict(
        dx=dx,
        N=N,
        c1=c1,
        nu=nu,
        k0=k0,
        poisson_bracket_coeff=poisson_bracket_coeff,
        kappa_coeff=kappa_coeff,
        zonal=zonal
    )
    # Initialize Plasma
    plasma = Namespace(
        density=noise[init_type](y, x) * init_scale,
        omega=noise[init_type](y, x) * init_scale,
        phi=noise[init_type](y, x) * init_scale,
        age=0,
    )

    # File Handling
    if output_path:
        y_save = y
        x_save = x
        if downsample_factor != 1:
            y_save = int(round(y / downsample_factor))
            x_save = int(round(x / downsample_factor))
            print(f"Downsample by {downsample_factor}x: {y}x{x}->{y_save}x{x_save}")
        # Output data from this run
        buffer = {
            field: np.zeros((buffer_length, y_save, x_save), dtype=np.float32)
            for field in field_list
        }
        # Property buffer
        buffer[f"time"] = np.zeros((buffer_length, 1), dtype=np.float32)
        for prop_name in properties:
            buffer[f"fullres_{prop_name}"] = np.zeros(
                (buffer_length, 1), dtype=np.float32
            )
        # Other output parameters
        output_params = {
            "buffer": buffer,
            "buffer_index": 0,
            "output_path": output_path,
        }
        # Load Data
        if os.path.isfile(output_path) and not force_recompute:
            if continue_file:
                plasma, physics_params = continue_h5_file(output_path, field_list)
                save_params = get_save_params(
                    physics_params,
                    step_size,
                    snaps,
                    x,
                    y,
                    x_save=x_save,
                    y_save=y_save,
                    recording_start_time=recording_start_time,
                    adaptive_step_size=adaptive_step_size,
                )
                initial_time = np.float64(plasma.age)
                print(
                    f"Loaded: {output_path} (age={plasma.age}, time={initial_time})\n{physics_params}"
                )
                # Check if broken
                for field in plasma.keys():
                    # Not NaNs
                    if np.isnan(np.sum(plasma[field])):
                        print(f"Input file is broken: NaNs in {field}")
                        raise BaseException(f"Input file is broken: NaNs in {field}")
                    # Not zeros
                    if field in ("density", "omega"):
                        if np.all(plasma[field] == 0):
                            print(f"FAILED {field} is zero")
                            raise BaseException(f"Input File is zero: {field}")
            else:
                print(f"File already exists.")
                return
        # Create Data
        else:
            save_params = get_save_params(
                physics_params,
                step_size,
                snaps,
                x,
                y,
                x_save=x_save,
                y_save=y_save,
                recording_start_time=recording_start_time,
                adaptive_step_size=adaptive_step_size,
            )
            # Create file
            create_appendable_h5(
                output_path,
                save_params,
                properties=["time"]
                + [f"fullres_{prop_name}" for prop_name in properties],
                dtype=np.float32,
                chunk_size=chunk_size,
                add_last_state=add_last_state,
            )
            # Save initial values
            if not recording_start_time:
                new_val = Namespace(
                    **{
                        k: v
                        for k, v in plasma.items()
                        if k in ("phi", "omega", "density")
                    }
                )
                last_state = {}
                for k, v in plasma.items():
                    if k in ("phi", "omega", "density"):
                        if downsample_factor != 1:
                            new_val[k] = downsample_fnc(v, downsample_factor)
                        if add_last_state:
                            last_state[f"state_{k}"] = v
                for prop_name in properties:
                    new_val[f"fullres_{prop_name}"] = property_fncs[prop_name](
                        n=plasma["density"],
                        p=plasma["phi"],
                        o=plasma["omega"],
                        dx=dx,
                        c1=c1,
                    )
                new_val["time"] = initial_time
                output_params["buffer_index"] = save_to_buffered_h5(
                    new_val=new_val,
                    last_state=last_state,
                    buffer_length=buffer_length,
                    **output_params,
                )

    # Display runtime parameters
    # save_params["output_path"] = output_path  # No support for strings
    format_print_dict(save_params)

    # Setup Simulation
    hw = HW(**physics_params, debug=debug)
    plasma["phi"] = hw.get_phi(plasma.omega, physics_params["dx"])

    # Setup Progress Bar
    print("Running simulation...")
    iteration_count = 0
    bar_format = "{rate_fmt} | {desc} {percentage:>6.2f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]"
    # bar_format = "{percentage:>6.2f}%|{bar}| {n:.4f}/{total_fmt} [{elapsed}<{remaining}] ({rate_fmt})"
    used_step_size = step_size
    used_time_steps = []
    current_time = initial_time
    new_val = {}

    try:
        with tqdm(
            total=end_time - initial_time,
            unit="t",
            bar_format=bar_format,
        ) as pbar:
            # Run Simulation
            while round(current_time, 4) < end_time:
                # Progress one step, alternatively: hw.euler_step()
                successful = False
                while not successful:
                    try:
                        plasma = hw.rk4_step(plasma, dt=step_size, dx=dx)
                        used_step_size = step_size
                        successful = True
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                iteration_count += 1

                # Improved accuracy of time tracking
                if adaptive_step_size:
                    used_time_steps.append(np.float64(used_step_size))
                    current_time = initial_time + np.sum(used_time_steps)
                else:
                    current_time = initial_time + iteration_count * step_size

                # Batched Saving
                # TODO: Figure out whether it should be iteration_count % (snaps - 1) == 0
                if (current_time >= recording_start_time) and (
                    iteration_count % snaps == 0
                ):
                    new_val = Namespace(
                        **{
                            k: v.copy()
                            for k, v in plasma.items()
                            if k in ("phi", "omega", "density")
                        }
                    )
                    # Add properties if they were selected
                    # TODO: Faster to do before saving through trivial parallelization
                    new_val["time"] = current_time
                    for prop_name in properties:
                        new_val[f"fullres_{prop_name}"] = property_fncs[prop_name](
                            n=plasma["density"],
                            p=plasma["phi"],
                            o=plasma["omega"],
                            dx=dx,
                            c1=c1,
                        )

                    # Save to records
                    if output_path:
                        last_state = {}
                        # Save downsampled & state
                        for k, v in plasma.items():
                            if k in ("phi", "omega", "density"):
                                # Downsample for saving
                                if downsample_factor != 1:
                                    new_val[k] = downsample_fnc(v.copy(), downsample_factor)
                                # Add last state as high-res for continuing simulation
                                if add_last_state:
                                    last_state[f"state_{k}"] = v
                        # Save to buffer/file
                        output_params["buffer_index"] = save_to_buffered_h5(
                            new_val=new_val,
                            last_state=last_state,
                            buffer_length=buffer_length,
                            **output_params,
                        )

                    # Check for breaking
                    if np.isnan(np.sum(plasma.density)):
                        print(f"FAILED @ {iteration_count:,} steps ({plasma.age:,})")
                        break

                # Calculate iterations per second, handling division by zero
                try:
                    iter_per_sec = iteration_count / pbar.format_dict["elapsed"]
                except ZeroDivisionError:
                    iter_per_sec = float("inf")  # or set to 0 or any default value

                # Update progress
                pbar.update(used_step_size)
                new_description = f"{iter_per_sec:.2f}it/s"
                if show_property:
                    new_description += f" | Γn = {new_val.get(show_property, 0):.2g}"
                pbar.set_description(new_description)

    except Exception as e:
        print(e)
        raise e
        # os.remove(output_path)
        # print(f"removed {output_path}")
        return

    # If output_path is defined, flush any remaining data in the buffer
    if (
        output_path
        and (current_time > recording_start_time)
        and (output_params["buffer_index"] > 0)
    ):
        append_h5(**output_params)

    # Get Performance stats
    hw.print_log()

    # Generate Movie from saved file
    if movie and output_path:
        print(f"Generating movie...")
        create_movie(
            input_filename=output_path,
            output_filename=output_path.replace(".h5", ""),
            t0=0,
            t1=None,
            plot_order=field_list,
            min_fps=min_fps,
            dpi=dpi,
            speed=speed,
        )

    if properties and output_path:
        print(f"Calculating properties...")
        calculate_properties(
            file_path=output_path,
            batch_size=buffer_length,
            property_list=properties,
            force_recompute=True,
            is_debug=False,
        )

    if plot_properties and output_path:
        print(f"Plotting properties...")
        plot_timetrace_comparison(
            file_path=output_path,
            out_path=None,
            properties=plot_properties,
            t0=0,
            t0_std=t0_std,
        )


if __name__ == "__main__":
    fire.Fire(run)
