import fire
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Local imports
from hw2d.model import HW  # (grid_size, L)
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
from hw2d.physical_properties.numpy_properties import (
    get_gamma_n_spectrally,
    get_gamma_n,
)


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


def run(
    step_size: float = 0.025,
    end_time: float = 100,
    grid_pts: int = 512,
    k0: float = 0.15,
    N: int = 3,
    nu: float = 5.0e-08,
    c1: float = 1.0,
    kappa_coeff: float = 1.0,
    arakawa_coeff: float = 1.0,
    seed: int or None = None,
    init_type: str = "normal",
    init_scale: float = 1 / 100,
    snaps: int = 40,
    buffer_size: int = 100,
    output_path: str = "",
    continue_file: bool = False,
    movie: bool = False,
    min_fps: int = 10,
    dpi: int = 75,
    speed: float = 5,
    debug: bool = False,
):
    """
    Run the simulation with the given parameters.

    Parameters:
    - step_size (float): Incremental step for simulation progression. Default is 0.025.
    - end_time (float): Duration till the simulation should run. Default is 5.
    - grid_pts (int): Grid points. Suggested: 128 for coarse, 1024 for fine. Default is 512.
    - k0 (float): Determines k-focus. Suggested: 0.15 for high-k, 0.0375 for low-k. Default is 0.15.
    - N (int): Dissipation exponent's half value. Default is 3.
    - nu (float): Viscosity. Suggested: 5e-10 for coarse-large, 1e-4 for fine-small. Default is 1e-08.
    - c1 (float): Transition scale between hydrodynamic and adiabatic. Suggested values: 0.1, 1, 5. Default is 1.0.
    - kappa_coeff, arakawa_coeff (float): Coefficients for simulation. Default is 1.0 for both.
    - seed (int): Seed for random number generation. Default is None.
    - init_type (str): Initialization method. Choices: 'fourier', 'sine', 'random', 'normal'. Default is 'normal'.
    - init_scale (float): Scaling factor for initialization. Default is 0.01.
    - snaps (int): Snapshot intervals for saving. Default is 1.
    - buffer_size (int): Size of buffer for storage. Default is 100.
    - output_path (str): Where to save the simulation data. Default is current directory with filename 'test.h5'.
    - continue_file (bool): If True, continue with existing file. Default is True.
    - movie (bool): If True, generate a movie out of simulation. Default is False.
    - min_fps, dpi, speed (int): Parameters for movie generation. Default values are 5, 75, 5 respectively.
    - debug (bool): Enable or disable debug mode. Default is False.

    Returns:
    None. The function saves simulation data or generates a movie as specified.
    """

    global noise

    # Unpacking
    y = grid_pts
    x = grid_pts
    L = 2 * np.pi / k0  # Physical box size
    dx = L / x  # Grid resolution
    steps = int(end_time / step_size)  # Number of Steps until end_time
    snap_count = steps // snaps + 1  # number of snapshots
    field_list = ("density", "omega", "phi")
    np.random.seed(seed)

    # Physics
    physics_params = dict(
        dx=dx,
        N=N,
        c1=c1,
        nu=nu,
        k0=k0,
        arakawa_coeff=arakawa_coeff,
        kappa_coeff=kappa_coeff,
    )
    # Initialize Plasma
    plasma = Namespace(
        density=noise[init_type](y, x) * init_scale,
        omega=noise[init_type](y, x) * init_scale,
        phi=noise[init_type](y, x) * init_scale,
        age=0,
        dx=dx,
    )

    # File Handling
    if output_path:
        buffer = {
            field: np.zeros((buffer_size, y, x), dtype=np.float32)
            for field in field_list
        }
        output_params = {
            "buffer": buffer,
            "buffer_index": 0,
            "output_path": output_path,
        }
        # Load Data
        if os.path.isfile(output_path):
            if continue_file:
                plasma, physics_params = continue_h5_file(output_path, field_list)
                print(
                    f"Successfully loaded: {output_path} (age={plasma.age})\n{physics_params}"
                )
            else:
                print(f"File already exists.")
                return
        # Create
        else:
            save_params = get_save_params(physics_params, step_size, snaps, x, y)
            create_appendable_h5(
                output_path, save_params, dtype=np.float32, chunk_size=100
            )
            output_params["buffer_index"] = save_to_buffered_h5(
                new_val=plasma, buffer_size=buffer_size, **output_params
            )

    # Setup Simulation
    hw = HW(**physics_params, debug=debug)
    plasma["phi"] = hw.get_phi(plasma.omega, physics_params["dx"])

    # Run Simulation
    for i in tqdm(range(1, steps + 1)):
        plasma = hw.euler_step(plasma, dt=step_size, dx=dx)
        plasma = hw.rk4_step(plasma, dt=step_size, dx=dx)
        # if i % (40 * 10) == 0:
        #     gamma_n_spectral = get_gamma_n_spectrally(
        #         plasma["density"], plasma["phi"], dx=dx
        #     )
        #     gamma_n = get_gamma_n(plasma["density"], plasma["phi"], dx=dx)
        #     print(f"{plasma.age:>8.3f}  {gamma_n:>8.2e} {gamma_n_spectral:>8.2e}")
        # elif i % (40 * 2) == 0:
        #     gamma_n = get_gamma_n(plasma["density"], plasma["phi"], dx=dx)
        #     print(f"{plasma.age:>8.3f}  {gamma_n:>8.2e}")

        # Save to records
        if output_path and i % snaps == 0:
            output_params["buffer_index"] = save_to_buffered_h5(
                new_val=plasma, buffer_size=buffer_size, **output_params
            )

        # Check for breaking
        if np.isnan(np.sum(plasma.density)):
            print(f"FAILED @ {i:,} steps ({plasma.age:,})")
            break

    # If output_path is defined, flush any remaining data in the buffer
    if output_path and output_params["buffer_index"] > 0:
        append_h5(**output_params)

    # Get Performance stats
    hw.print_log()

    # Generate Movie
    if movie and output_path:
        create_movie(
            input_filename=output_path,
            output_filename=output_path,
            t0=0,
            t1=None,
            plot_order=field_list,
            min_fps=min_fps,
            dpi=dpi,
            speed=speed,
        )


if __name__ == "__main__":
    fire.Fire(run)
