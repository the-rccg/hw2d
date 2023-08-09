import numpy
import h5py
import fire
from tqdm import tqdm
from typing import List

from hw2d.physical_properties.numpy_properties import *
from hw2d.utils.io import *


def main(
    file_path: str = "/ptmp/rccg/hw2d/512x512_dt=0.025_nu=1.0e-10.h5",
    t0: int = 0,
    t1: int = 0,
    batch_size=100,
    property_list: List = [
        "gamma_n",
        "gamma_n_spectral",
        "gamma_c",
        "energy",
        "enstrophy",
        "enstrophy_phi",
    ],
    force_recompute: bool = False,
    is_debug: bool = False,
):
    with h5py.File(file_path, "r+") as h5_file:
        parameters = dict(h5_file.attrs)
        steps = len(h5_file["density"])
        # Create Properties in Dataet
        property_run_list = []
        for property_name in property_list:
            if property_name in h5_file.keys():
                print(f"Dataset exists:  {property_name}")
                if force_recompute:
                    del h5_file[property_name]
            if property_name not in h5_file.keys():
                property_run_list.append(property_name)
                h5_file.create_dataset(property_name, (steps,))
                print(f"Created Dataset:  {property_name}")
        # Run Through in Batches
        for i in tqdm(range(0, steps, batch_size)):
            n = h5_file["density"][i : i + batch_size]
            p = h5_file["phi"][i : i + batch_size]
            o = h5_file["omega"][i : i + batch_size]
            gamma_n = get_gamma_n(n, p, dx=parameters["dx"])
            h5_file["gamma_n"][i : i + batch_size] = gamma_n
            gamma_c = get_gamma_c(n, p, c1=parameters["c1"], dx=parameters["dx"])
            h5_file["gamma_c"][i : i + batch_size] = gamma_c
            gamma_n_spectral = [
                get_gamma_n_spectrally(n[i], p[i], dx=parameters["dx"])
                for i in range(len(n))
            ]
            h5_file["gamma_n_spectral"][i : i + batch_size] = gamma_n_spectral
            energy = get_energy(n, phi=p, dx=parameters["dx"])
            h5_file["energy"][i : i + batch_size] = energy
            enstrophy = get_enstrophy(n=n, omega=o, dx=parameters["dx"])
            h5_file["enstrophy"][i : i + batch_size] = enstrophy
            enstrophy_phi = get_enstrophy_phi(n=n, phi=p, dx=parameters["dx"])
            h5_file["enstrophy_phi"][i : i + batch_size] = enstrophy_phi
            # Debug printing
            if is_debug:
                print(
                    f"Timeframe ({i*parameters['dt']}-{(i + batch_size)*parameters['dt']})"
                )
                print(
                    "  |  ".join(
                        [
                            f"  Gamma_n: {np.mean(gamma_n):.2e}",
                            f"Gamma_n(k): {np.mean(gamma_n_spectral):.2e}",
                        ]
                    )
                )
                print(f"  Gamma_c: {np.mean(gamma_c):2e}")
                print(f"  Energy: {np.mean(energy):2e}")
                print(
                    "  |  ".join(
                        [
                            f"  Enstrophy: {np.mean(enstrophy):.2e}",
                            f"Enstrophy(Phi): {np.mean(enstrophy_phi):.2e}",
                        ]
                    )
                )

    return


if __name__ == "__main__":
    fire.Fire(main)
