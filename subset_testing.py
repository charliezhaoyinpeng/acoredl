"""Utility to create ground truth for open set experiments."""

import argparse
import os
import csv
import pandas as pd
import pickle as pkl
from calc_mAP import read_labelmap, read_csv, read_exclusions
from typing import Dict, List
from io import TextIOWrapper


def cmd_opts() -> None:
    """
    Command line for the utility.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser("Utility to create ground truth files for evaluation")
    parser.add_argument("--labelmap-path",
                        type=argparse.FileType("r"),
                        help="Path to labelmap used in evaluation")
    parser.add_argument("--gt-csv-path",
                        type=argparse.FileType("r"),
                        help="Path to ground truth csv")
    parser.add_argument("--exclusion-timestamp-path",
                        type=argparse.FileType("r"),
                        help="Path to file with exclusion timestamp")
    parser.add_argument("--output-directory",
                        help="Path to a directory where updated GT files are written")
    args = parser.parse_args()
    return args


def write_labelmap(labelmap_entries: List[Dict], labelmap_path: str) -> None:
    """
    Helper function to write labelmap.

    Args:
        labelmap_entries: A list of dictionary with id and name of a class
        labelmap_path: Path where labelmap is written

    Returns:
        None
    """
    with open(labelmap_path, "w") as lp:
        for labelmap_entry in labelmap_entries:
            lp.write(f'item {{{os.linesep}')
            lp.write(f'  name: \"{labelmap_entry["name"]}\"{os.linesep}')
            lp.write(f'  id: {labelmap_entry["id"]}{os.linesep}}}{os.linesep}')


def write_filtered_gt_csv(
        filtered_boxes: Dict[List],
        filtered_labels: Dict[List],
        gt_csv: TextIOWrapper,
        gt_csv_path: str) -> None:
    """
    Write ground truth filtered for open set evaluation.

    Args:
        filtered_boxes: Dictionary with bounding boxes
        filtered_labels: Dictionary with labels
        gt_csv: Text wrapper for ground truth
        gt_csv_path: Path where filtered gt csv is written

    Returns:
        None
    """
    pid_dict = {}
    for row_entries in csv.reader(gt_csv):
        # PID is the last entry in the annotation, we are going to use all the
        # other entries in the annotation as key
        pid_dict[",".join(row_entries[:-1])] = row_entries[-1]
    with open(gt_csv_path, "w") as gtp:
        for fbox_id, fboxes in filtered_boxes.items():
            labels = filtered_labels[fbox_id]
            for fbox, label in zip(fboxes, labels):
                y1, x1, y2, x2 = fbox
                pid_key = f"{fbox_id},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{label}"
                pid = pid_dict[pid_key]
                gtp.write(f"{pid_key},{pid}{os.linesep}")


def main(
        labelmap: TextIOWrapper,
        gt_csv: TextIOWrapper,
        exclusion_timestamp: TextIOWrapper,
        output_directory: str) -> None:
    """
    Main function for the utility.

    Args:
        labelmap: Text wrapper for labelmap
        gt_csv: Text wrapper for ground truth
        exclusion_timestamp: Text wrapper for exclusion timestamp file
        output_directory: Directory where the filtered ground truth is saved

    Returns:
        None
    """
    categories, class_whitelist = read_labelmap(labelmap)
    class_whitelist = sorted(list(class_whitelist))
    excluded_keys = read_exclusions(exclusion_timestamp)
    # TODO: Use cli parameter for then indices
    test_class_idx = list(range(20, 60))
    filtered_class_whitelist = []
    for idx, class_id in enumerate(class_whitelist):
        if idx in test_class_idx:
            filtered_class_whitelist.append(class_id)
    filtered_categories = []
    for category in categories:
        if category["id"] in filtered_class_whitelist:
            filtered_categories.append(category)

    boxes, labels, _ = read_csv(gt_csv, class_whitelist)
    print(f"Annotations before filtering: {len(boxes)}")
    gt_csv.seek(0)
    filtered_boxes, filtered_labels, _ = read_csv(gt_csv, filtered_class_whitelist)
    print(f"Annotations after filtering: {len(filtered_boxes)}")
    # Find set difference between labels and filtered labels
    diff_labels = set(labels.keys()) - set(filtered_labels.keys())

    # Add original exclusion in diff labels
    diff_labels.update(excluded_keys)

    os.makedirs(output_directory, exist_ok=True)

    timestamp_path = os.path.join(output_directory,
                                  "ava_val_excluded_timestamps_v2.2_openset.csv")
    labelmap_path = os.path.join(output_directory,
                                 "ava_action_list_v2.2_for_openset.pbtxt")
    gt_path = os.path.join(output_directory,
                           "ava_val_v2.2_openset.csv")

    with open(timestamp_path, "w") as tp:
        tp.writelines(map(lambda x: x + "\n", list(diff_labels)))

    write_labelmap(filtered_categories, labelmap_path)

    # Reset GT csv
    gt_csv.seek(0)
    write_filtered_gt_csv(filtered_boxes, filtered_labels, gt_csv, gt_path)


if __name__ == "__main__":
    args = cmd_opts()
    main(args.labelmap_path,
         args.gt_csv_path,
         args.exclusion_timestamp_path,
         args.output_directory)
