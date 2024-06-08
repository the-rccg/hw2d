import numpy as np
import h5py
import fire
from tqdm import tqdm
from typing import List, Iterable
from functools import partial

from hw2d.physical_properties.numpy_properties import (
    get_gamma_n,
    get_gamma_n_spectrally,
    get_gamma_c,
    get_energy,
    get_energy_N_spectrally,
    get_energy_V_spectrally,
    get_enstrophy,
    get_enstrophy_phi,
)


def add_data(hf: h5py.File, i: int, batch_size: int, name: str, data: np.ndarray, debug: bool, selection: Iterable[str]) -> None:
    if name in selection:
        data = data()
        hf[name][i : i + batch_size] = data
        if debug:
            print(f"{name}: {np.mean(data):.2e}", end="  |  ")


def calculate_properties(
    file_path: str = "",
    batch_size: int = 100,
    property_list: Iterable[str] = [
        "gamma_n",
        "gamma_n_spectral",
        "gamma_c",
        "energy",
        "thermal_energy",
        "kinetic_energy",
        "enstrophy",
        "enstrophy_phi",
    ],
    force_recompute: bool = True,
    is_debug: bool = False,
):
    with h5py.File(file_path, "r+") as h5_file:
        # Parameters
        parameters = dict(h5_file.attrs)
        L = 2 * np.pi / parameters["k0"]
        dx = L / parameters["x_save"]
        c1 = parameters["c1"]
        dt = parameters["frame_dt"]
        steps = len(h5_file["density"])
        # Create Properties in Dataset
        selection = []
        existing_datasets = []
        created_datasets = []
        for property_name in property_list:
            if property_name in h5_file.keys():
                existing_datasets.append(property_name)
                if force_recompute:
                    del h5_file[property_name]
            if property_name not in h5_file.keys():
                selection.append(property_name)
                h5_file.create_dataset(property_name, (steps,))
                created_datasets.append(property_name)
        print(f"Existing datasets:  {existing_datasets}")
        print(f"Created datasets:   {created_datasets}")
        # Set up iterator
        iterator = range(0, steps, batch_size)
        if not is_debug:
            iterator = tqdm(iterator)
        # Run Through in Batches
        for i in iterator:
            # Define addition function
            add = partial(
                add_data,
                hf=h5_file,
                i=i,
                batch_size=batch_size,
                debug=is_debug,
                selection=selection,
            )
            # Get intput data
            n = h5_file["density"][i : i + batch_size]
            p = h5_file["phi"][i : i + batch_size]
            o = h5_file["omega"][i : i + batch_size]
            if is_debug:
                print(
                    f"Timeframe ({i*dt}-{(i + batch_size)*dt})",
                    end="\n  ",
                )
            # Add properties if they were selected
            property_mapping = {
                "gamma_n": partial(get_gamma_n, n=n, p=p, dx=dx),
                "gamma_n_spectral": partial(get_gamma_n_spectrally, n=n, p=p, dx=dx),
                "gamma_c": partial(get_gamma_c, n=n, p=p, c1=c1, dx=dx),
                "energy": partial(get_energy, n=n, phi=p, dx=dx),
                "thermal_energy": partial(get_energy_N_spectrally, n=n),
                "kinetic_energy": partial(get_energy_V_spectrally, p=p, dx=dx),
                "enstrophy": partial(get_enstrophy, n=n, omega=o, dx=dx),
                "enstrophy_phi": partial(get_enstrophy_phi, n=n, phi=p, dx=dx),
            }
            for name, data in property_mapping.items():
                add(name=name, data=data)

    return


if __name__ == "__main__":
    fire.Fire(calculate_properties)
