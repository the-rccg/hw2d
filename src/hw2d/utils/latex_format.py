names = {
    "int_gamma_n": "$\\Gamma_n(t) = \\int{dk_y \\; \\Gamma_n \\left( k_y \\right) }$",
    "integrated gamma_n": "$\\Gamma_n(t) = \\int{dk_y \\; \\Gamma_n \\left( k_y \\right) }$",
    "gamma_n": "$\\Gamma_n$",
    "gamma_n_k": "$\\Gamma_n \\left(k_y\\right)$",
    "gamma_c": "$\\Gamma_c$",
    "ky_phi": "$\\phi \\left( k_y \\right)$",
    "ky_omega": "$\\Omega \\left( k_y \\right)$",
    "ky_density": "$n \\left( k_y \\right)$",
    "ky": "$k_y$",
    "mean": "$\\mu$",
    "std": "$\\sigma$",
    "delta_k": "$\\delta_k$",
    "U_phi": "$U_\\phi$",
    "U_centered": "$U \\left( \\Omega-\\mu \\left( \\Omega \\right), n \\right)$",
    "nu": "$\\nu$",
    "density": "density ($n$)",
    "phi": "potential ($\\phi$)",
    "omega": "vorticity ($\\Omega$)",
    "dE_dt": "$\\partial_t E$",
    "dU_dt": "$\\partial_t U$",
    "DE_v": "$DE_v$",
    "DE_n": "$DE_n$",
    "DEv": "$DE_v$",
    "DEn": "$DE_n$",
    "DE": "$DE$",
    "DU": "$DU$",
    "E": "E",
    "U": "U",
    "cfl": "cfl",
    "mean_gamma_n": "$\\mu \\left( \\Gamma_n \\right)$",
    "std_gamma_n": "$\\sigma \\left( \\Gamma_n \\right)$",
    "mean_gamma_c": "$\\mu \\left( \\Gamma_c \\right)$",
    "std_gamma_c": "$\\sigma \\left( \\Gamma_c \\right)$",
    "mean_int_gamma_n": "$\\mu_t \\left( \\int \\Gamma_n \\left(k_y\\right) \\right)$",
    "std_gamma_n": "$\\sigma \\left( \\int \\Gamma_n \\left(k_y\\right) \\right)$",
    "c1": "$c_1$",
    "kinetic_energy": "$E_K$",
    "thermal_energy": "$E_T$",
    "energy": "Energy $(E)$",
    "enstrophy": "Enstrophy $(U)$",
}


def metric_format(string: str) -> str:
    if any([s in string for s in ["mean", "median", "std"]]):
        return string[: string.find("_")] + " " + names[string[string.find("_") + 1 :]]
    return ""


def latex_format(string: str) -> str:
    try:
        return names[string]
    except Exception as e:
        print(string, e)
        return string
