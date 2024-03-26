import os
from typing import Dict
import numbers
import numpy as np


def is_number(value):
    # Check for basic Python numbers, NumPy numbers
    if isinstance(value, numbers.Number) or np.isscalar(value):
        return True
    return False


def format_print_dict(
    dictionary: Dict[str, float],
    spacing: str = " ",
    key_spacing: str = " | ",
    max_line_length: int = 90,
    value_format: str = ".2g",
    value_length: int = 7,
):
    """Print a dictionary with aligned values

    Args:
        dictionary (Dict): dictionary to be printed
        spacing (str, optional): Spacing between key and value. Defaults to " ".
        key_spacing (str, optional): Spacing within a row between keys. Defaults to " | ".
        max_line_length (int, optional): maximum line length for formatting. Defaults to 90.
        value_format (str, optional): Format to write values as. Defaults to ".2g".
        value_length (int, optional): Length of values reserved for the floating representation. Defaults to 7.
    """
    dictionary = {k: v for k,v in dictionary.items() if is_number(v)}
    key_lengths = [len(k) for k in dictionary.keys()]
    longest_key_length = max(key_lengths) + 1
    try:
        max_line_length = min(max_line_length, os.get_terminal_size().columns)
    except:
        pass
    # Iteratively match them and adjust length until it exceeds the terminal length
    # each column should be aligned
    keys = tuple(dictionary.keys())
    # Iterate from all in one line to multiple lines
    for i in range(len(keys), 0, -1):
        # Assign entries to columns
        columns_per_row = {j: keys[j::i] for j in range(0, i)}
        line_length = 0
        column_key_lengths = {}
        # Determine column widths
        for column_idx, key_pair in columns_per_row.items():
            key_lengths = [len(k) for k in key_pair]
            longest_key_length = max(key_lengths) + 1
            column_key_lengths[column_idx] = longest_key_length
            column_length = longest_key_length + len(spacing) + value_length
            line_length += column_length
        line_length += (i - 1) * len(key_spacing)
        # Check if it fits into space
        if line_length < max_line_length:
            # Generate each line
            number_of_lines = max([len(i) for i in columns_per_row.values()])
            lines = [[] for _ in range(number_of_lines)]  # [] for _ in range(len())]
            for column_idx in columns_per_row.keys():
                # Each entry in the column idx is a row
                for line_idx, k in enumerate(columns_per_row[column_idx]):
                    lines[line_idx].append(
                        (
                            f"{k+':':<{column_key_lengths[column_idx]}}"
                            + f"{spacing}"
                            + f"{dictionary[k]:>{value_length}{value_format}}"
                        )
                    )
            lines = [key_spacing.join(l) for l in lines]
            # print(f"number of columns: {i}  |  {line_length=}  |  {max([len(l) for l in lines])=}  |  {number_of_lines=}")
            lines = "\n".join(lines)
            print(lines)
            return
