import numpy as np
import perfplot as pp
import fire
from typing import List


# Function definitions
def setup_random_array(n, scale=1000):
    return np.random.rand(n, n) / scale

def setup_random_arrays_pair(n, scale=1000):
    return [setup_random_array(n, scale), setup_random_array(n, scale)]

def get_function_defs(function_name):
    if function_name == "periodic_gradient":
        from hw2d.gradients.numpy_gradients import periodic_gradient as np_periodic_gradient
        from hw2d.gradients.numba_gradients import periodic_gradient as nb_periodic_gradient
        np_instrinsic_gradient = lambda x, dx: np.gradient(np.pad(x, 1, mode="wrap"), axis=-2)[1:-1, 1:-1]
        return [
            ("np_periodic_gradient", np_periodic_gradient),
            ("np_instrinsic_gradient", np_instrinsic_gradient),
            ("nb_periodic_gradient", nb_periodic_gradient),
        ]
    elif function_name == "periodic_laplace":
        from hw2d.gradients.numpy_gradients import periodic_laplace as np_periodic_laplace_N
        from hw2d.gradients.numba_gradients import periodic_laplace as nb_periodic_laplace_N
        from hw2d.gradients.numpy_gradients import fourier_laplace as np_fourier_laplace
        return [
            ("np_periodic_laplace", np_periodic_laplace_N),
            #("np_fourier_laplace", np_fourier_laplace),
            ("nb_periodic_laplace", nb_periodic_laplace_N),
        ]
    elif function_name == "poisson_solvers":
        from hw2d.poisson_solvers.numpy_fourier_poisson import fourier_poisson_double as np_fourier_poisson_double
        from hw2d.poisson_solvers.numba_fourier_poisson import fourier_poisson_double as nb_fourier_poisson_double
        return [
            ("np_fourier_poisson_double", np_fourier_poisson_double),
            ("nb_fourier_poisson_double", nb_fourier_poisson_double),
        ]
    elif function_name == "poisson_bracket":
        from hw2d.poisson_bracket.numpy_arakawa import periodic_arakawa as np_arakawa
        from hw2d.poisson_bracket.numpy_arakawa import periodic_arakawa_vec as np_arakawa_vec
        from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa as nb_arakawa
        from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa_stencil as nb_arakawa_stencil
        from hw2d.poisson_bracket.numba_arakawa import periodic_arakawa_vec as nb_arakawa_vec
        return [
            ("np_arakawa", np_arakawa),
            ("np_arakawa_vec", np_arakawa_vec),
            ("nb_arakawa", nb_arakawa),
            ("nb_arakawa_stencil", nb_arakawa_stencil),
            ("nb_arakawa_vec", nb_arakawa_vec)
        ]

# Main function
def main(
    max_time: int = 30,  # seconds
    scale: int = 1000,
    dx: float = 1,
    function_names: List[str] = ["periodic_gradient", "periodic_laplace", "poisson_bracket", "poisson_solvers"],  # Add or remove function names as needed
):
    sizes = 2 ** np.array(range(4, 13))
    for function_name in function_names:
        # Get function definitions
        function_defs = get_function_defs(function_name)
        # Repack to labels and kernels
        labels = [t[0] for t in function_defs]
        kernels = [t[1] for t in function_defs]
        # Setup functions for benchmark
        if function_name != "poisson_bracket":
            kernels = [lambda a, f=f: f(a, dx=dx) for f in kernels]
            setup_fnc = setup_random_array
        else:
            kernels = [lambda ab, f=f: f(*ab, dx=dx) for f in kernels]
            setup_fnc = setup_random_arrays_pair
        # Bechmark
        benchmark = pp.bench(
            setup=setup_fnc,
            kernels=kernels,
            labels=labels,
            n_range=sizes,
            xlabel="len(a)",
            max_time=max_time,
        )
        # Plot
        #benchmark.show()
        print(benchmark)

if __name__ == "__main__":
    fire.Fire(main)
