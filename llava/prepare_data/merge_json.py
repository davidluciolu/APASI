import os
import json
import argparse

def merge_json_files(prefix, chunks):
    # 查找所有匹配的文件
    json_files = [f'{prefix}_{str(chunk_id)}.json' for chunk_id in range(chunks)]

    merged_data = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                merged_data.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {json_file}: {e}")

    output_file = f"{prefix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(json_files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="facebook/opt-350m")

    args = parser.parse_args()

    merge_json_files(args.prefix, args.num_chunks)
