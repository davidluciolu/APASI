import os
import pandas as pd
import argparse

def merge_parquet_files(prefix, chunks):

    parquet_files = [f'{prefix}_{str(chunk_id)}.parquet' for chunk_id in range(chunks)]
    print(parquet_files)
    dfs = [pd.read_parquet(file) for file in parquet_files]

    output_file = f"{prefix}.parquet"

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_parquet(output_file)

    print(f"Merged {len(parquet_files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="facebook/opt-350m")

    args = parser.parse_args()

    merge_parquet_files(args.prefix, args.num_chunks)
