import fire
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
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
from hw2d.utils.plot.timetrace import plot_timetraces
from hw2d.utils.downsample import fourier_downsample
from hw2d.utils.helpers import format_print_dict


def run(
    # Physics & Numerics
    step_size: float = 0.025,
    end_time: float = 1_000,
    grid_pts: int = 1024,
    k0: float = 0.15,
    N: int = 3,
    nu: float = 5.0e-08,
    c1: float = 1.0,
    kappa_coeff: float = 1.0,
    poisson_bracket_coeff: float = 1.0,
    # Initialization
    seed: int or None = None,
    init_type: str = "normal",
    init_scale: float = 1 / 100,
    # Saving
    output_path: str = "_test.h5",
    continue_file: bool or str = False,
    buffer_length: int = 100,
    snaps: int = 1,
    chunk_size: int = 100,
    downsample_factor: float = 1,
    add_last_state: bool = False,
    # Movie
    movie: bool = True,
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
    plot_properties: Iterable[str] = (
        "gamma_c",
        "gamma_n",
        "gamma_n_spectral",
        "enstrophy",
        "energy",
        "kinetic_energy",
        "thermal_energy",
    ),
    # Other
    debug: bool = False,
    force_recompute: bool = True,
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
        seed (int or None, optional): Seed for random number generation. Defaults to None.
        init_type (str, optional): Initialization method. Choices: 'fourier', 'sine', 'random', 'normal'. Defaults to 'normal'.
        init_scale (float, optional): Scaling factor for initialization. Defaults to 0.01.
        output_path (str, optional): Where to save the simulation data. Defaults to ''.
        continue_file (bool or str, optional): If True, continue with existing file. If path, it will continue with file from path. Defaults to False.
        buffer_length (int, optional): Size of buffer for storage. Defaults to 100.
        snaps (int, optional): Snapshot intervals for saving. Defaults to 1.
        chunks (int, optional): Chunks of the h5 file. Defaults to 100.
        downsample_factor (float, optional): Factor along each axis that it is downsampled with for saving. Defaults to 1.
        add_last_state (bool, optional): Whether the last high-resolution frame is saved. Turns True if downsampling on. Defaults to False.
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

    # Unpacking
    y = grid_pts
    x = grid_pts
    L = 2 * np.pi / k0  # Physical box size
    dx = L / x  # Grid resolution
    steps = int(end_time / step_size)  # Number of Steps until end_time
    snap_count = steps // snaps + 1  # number of snapshots
    field_list = ("density", "omega", "phi")
    current_time = 0
    iteration_count = 0
    np.random.seed(seed)
    if downsample_factor != 1:
        add_last_state = True

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
    physics_params = dict(
        dx=dx,
        N=N,
        c1=c1,
        nu=nu,
        k0=k0,
        poisson_bracket_coeff=poisson_bracket_coeff,
        kappa_coeff=kappa_coeff,
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
            print(f"Downsample by {downsample_factor}x to: {y_save}x{x_save}")
        # Output data from this run
        buffer = {
            field: np.zeros((buffer_length, y_save, x_save), dtype=np.float32)
            for field in field_list
        }
        # Property buffer
        for p in ["time"]:
            buffer[p] = np.zeros((buffer_length, 1), dtype=np.float32)
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
                print(f"Loaded: {output_path} (age={plasma.age})\n{physics_params}")
                save_params = get_save_params(
                    physics_params, step_size, snaps, x, y, x_save=x_save, y_save=y_save
                )
                current_time = plasma.age
                # Check consistency
                for field in plasma.keys():
                    # No NaNs
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
            # Initial Values
            new_val = Namespace(
                **{k: v for k, v in plasma.items() if k in ("phi", "omega", "density")}
            )
            last_state = {}
            for k, v in plasma.items():
                if k in ("phi", "omega", "density"):
                    if downsample_factor != 1:
                        new_val[k] = downsample_fnc(v, downsample_factor)
                    if add_last_state:
                        last_state[f"state_{k}"] = v
            new_val["time"] = current_time
            save_params = get_save_params(
                physics_params, step_size, snaps, x, y, x_save=x_save, y_save=y_save
            )
            # Create file
            create_appendable_h5(
                output_path,
                save_params,
                properties=["time"],
                dtype=np.float32,
                chunk_size=chunk_size,
                add_last_state=add_last_state,
            )
            # Save initial values
            output_params["buffer_index"] = save_to_buffered_h5(
                new_val=new_val,
                last_state=last_state,
                buffer_length=buffer_length,
                **output_params,
            )

    # Display runtime parameters
    format_print_dict(save_params)

    # Setup Simulation
    hw = HW(**physics_params, debug=debug)
    plasma["phi"] = hw.get_phi(plasma.omega, physics_params["dx"])

    # Run Simulation
    print("Running simulation...")
    for iteration_count in tqdm(range(1, steps + 1)):
        # Progress one step, alternatively: hw.euler_step()
        plasma = hw.rk4_step(plasma, dt=step_size, dx=dx)

        # Save to records
        if output_path and iteration_count % snaps == 0:
            new_val = Namespace(
                **{k: v for k, v in plasma.items() if k in ("phi", "omega", "density")}
            )
            last_state = {}
            new_val["time"] = current_time
            # Save downsampled & state
            for k, v in plasma.items():
                if k in ("phi", "omega", "density"):
                    if downsample_factor != 1:
                        new_val[k] = downsample_fnc(v, downsample_factor)
                    if add_last_state:
                        last_state[f"state_{k}"] = v
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

    # If output_path is defined, flush any remaining data in the buffer
    if output_path and output_params["buffer_index"] > 0:
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
        plot_timetraces(
            file_path=output_path,
            out_path=None,
            properties=plot_properties,
            t0=0,
            t0_std=300,
        )


if __name__ == "__main__":
    fire.Fire(run)
