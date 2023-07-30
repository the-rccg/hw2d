import h5py
import numpy as np
from typing import Dict, List, Tuple, Any
from hw2d.utils.namespaces import Namespace


def get_save_params(params, dt, snaps, x, y):
    params = params.copy()
    params["dt"] = dt
    params["frame_dt"] = dt * snaps
    params["x"] = x
    params["y"] = y
    params["grid_pts"] = x
    return params


def create_appendable_h5(filepath, params, dtype=np.float32, chunk_size=100, field_list=["density", "omega", "phi"]):
    y = params["y"]
    x = params["x"]
    with h5py.File(f"{filepath}", "w") as hf:
        for field_name in field_list:
            hf.create_dataset(
                field_name,
                dtype=dtype,
                shape=(0, y, x),
                maxshape=(None, y, x),
                chunks=(chunk_size, y, x),
                compression="gzip",
            )
        for key, value in params.items():
            hf.attrs[key] = value


def append_h5(output_path, buffer, buffer_index):
    """append a file, from buffer, with buffer_index size"""
    with h5py.File(output_path, "a") as hf:
        for field_name in buffer.keys():
            _ = hf[field_name].resize((hf[field_name].shape[0] + buffer_index), axis=0)
            hf[field_name][-buffer_index:] = buffer[field_name][:buffer_index]


def save_to_buffered_h5(buffer: Dict[str, Any], buffer_size: int, buffer_index: int, new_val: Dict[str, Any], output_path: str, field_list: List[str] = ["density", "omega", "phi"]) -> int:
    """
    Save data to a buffer. If the buffer is full, flush the buffer to the HDF5 file.

    Args:
        buffer (Dict[str, Any]): Data buffer.
        buffer_size (int): Maximum size of the buffer.
        new_val (Dict[str, Any]): New values to be added to the buffer.
        field_list (List[str]): List of fields to be saved.
        buffer_index (int): Current index in the buffer.
        flush_index (int): Index to start flushing in the HDF5 file.
        output_path (str): Path of the output HDF5 file.

    Returns:
        Tuple[int, int]: Updated buffer index and flush index.
    """
    for idx, field in enumerate(field_list):
        buffer[field][buffer_index] = new_val[field]
    buffer_index += 1
    # If buffer is full, flush to HDF5 and reset buffer index
    if buffer_index == buffer_size:
        append_h5(output_path, buffer, buffer_index)
        buffer_index = 0
    return buffer_index


def load_h5_data(file_name: str, field_list: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load data and attributes from an HDF5 file.

    Args:
        file_name (str): Name and path of the HDF5 file to load.
        field_list (List[str]): List of fields to be loaded.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Loaded data and associated parameters (attributes).
    """
    with h5py.File(file_name, "r") as h5_file:
        data = {}
        for field in field_list:
            data[field] = h5_file[field][:]
        params = dict(h5_file.attrs)
    return data, params


def continue_h5_file(file_name: str, field_list: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load data and attributes from an HDF5 file.

    Args:
        file_name (str): Name and path of the HDF5 file to load.
        field_list (List[str]): List of fields to be loaded.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Loaded data and associated parameters (attributes).
    """
    lengths = []
    with h5py.File(file_name, "r") as h5_file:
        data = {}
        for field in field_list:
            data[field] = h5_file[field][-1].astype(np.float64)
            lengths.append(len(h5_file[field]))
        params = dict(h5_file.attrs)
    length = min(lengths)
    age = params["frame_dt"] * (length-1)
    data = Namespace(**data, age=age, dx=params["dx"])
    params = {k:params[k] for k in ("dx", "N", "c1", "nu", "k0", "arakawa_coeff", "kappa_coeff")}
    return data, params
