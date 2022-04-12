# acoredl

## Generating Ground Truth For Evaluation

Use `subset_testing.py` with the following parameters

```
python subset_testing.py --labelmap-path annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt \
                         --gt-csv-path annotations/ava_val_v2.2.csv \
                         --exclusion-timestamp-path annotations/ava_val_excluded_timestamps_v2.2.csv \
                         --output-directory openset-annotations/
```
This would create the ground truth files in openset-annotations folder
