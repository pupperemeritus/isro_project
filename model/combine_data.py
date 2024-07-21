import os
import sys

import polars as pl


def combine_data_into_csv(
    csv_directory: str = sys.argv[1], output_file: str = sys.argv[2]
):
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]

    # Read and concatenate CSV files
    dfs = [pl.read_csv(os.path.join(csv_directory, file)) for file in csv_files]
    concatenated_df = pl.concat(dfs)

    # Write to Arrow IPC format
    concatenated_df.write_ipc(output_file)

    print(f"Concatenated CSV files and exported to {output_file}")


if __name__ == "__main__":
    combine_data_into_csv(sys.argv[1], sys.argv[2])
