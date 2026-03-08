#!/usr/bin/env python3

import os
import traceback

from transforming_scripts.global_ranking_json_mine import convert_ranking


INPUT_FOLDER = "full_partition_jsons/partitions4/"
OUTPUT_FOLDER = "partition_txt/4/"
OVERWRITE = False


def main():
    input_folder = os.path.abspath(INPUT_FOLDER)
    output_folder = os.path.abspath(OUTPUT_FOLDER)

    if not os.path.isdir(input_folder):
        print(f"Error: input folder does not exist or is not a directory: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".json")
    )

    if not json_files:
        print(f"No .json files found in: {input_folder}")
        return

    print(f"Input folder : {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(json_files)} json file(s)\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for json_name in json_files:
        json_path = os.path.join(input_folder, json_name)
        txt_name = os.path.splitext(json_name)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_name)

        if os.path.exists(txt_path) and not OVERWRITE:
            print(f"[SKIP] {json_name} -> {txt_name} already exists")
            skip_count += 1
            continue

        print(f"[RUN ] {json_name} -> {txt_name}")
        try:
            convert_ranking(json_path, txt_path)
            print(f"[ OK ] Saved to {txt_path}\n")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {json_name}: {e}")
            traceback.print_exc()
            print()
            fail_count += 1

    print("Done.")
    print(f"  Successes: {success_count}")
    print(f"  Skipped  : {skip_count}")
    print(f"  Failed   : {fail_count}")


if __name__ == "__main__":
    main()