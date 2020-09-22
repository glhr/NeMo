#!/usr/bin/env bash

python fastsgt_predict.py --data_dir=/home/robotlab/dstc8-schema-guided-dialogue --no_overwrite_schema_emb_files --tracker_model=nemotracker --checkpoint_dir=/home/robotlab/NeMo/examples/nlp/dialogue_state_tracking/output/SGD/fastsgt/checkpoints --no_time_to_log_dir --debug_mode
