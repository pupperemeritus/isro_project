import os
import sys


import polars as pl


def find_csv_files(directory: str):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def combine_data_into_ipc(
    csv_directory: str = sys.argv[1], output_file: str = sys.argv[2]
):
    # Get all CSV files in the directory
    csv_files = find_csv_files(csv_directory)

    # Read and concatenate CSV files
    dfs = [pl.read_csv(os.path.join(csv_directory, file)) for file in csv_files]
    concatenated_df = pl.concat(dfs)

    # Write to Arrow IPC format
    concatenated_df.write_ipc(output_file)

    print(concatenated_df)

    print(f"Concatenated CSV files and exported to {output_file}")


def combine_data_into_parquet(
    csv_directory: str = sys.argv[1], output_file: str = sys.argv[2]
):
    # Get all CSV files in the directory
    csv_files = find_csv_files(csv_directory)

    # Read and concatenate CSV files
    dfs = [pl.read_csv(os.path.join(csv_directory, file)) for file in csv_files]
    concatenated_df = pl.concat(dfs)
    print(concatenated_df.head())
    # Write to Arrow IPC format
    concatenated_df.write_parquet(output_file)

    print(f"Concatenated CSV files and exported to {output_file}")


if __name__ == "__main__":
    if sys.argv[3] == "ipc":
        combine_data_into_ipc(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == "parquet":
        combine_data_into_parquet(sys.argv[1], sys.argv[2])
