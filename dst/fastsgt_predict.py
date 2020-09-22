# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import copy
import json
import os
import pickle
import re
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import argparse
import math
import os
import glob
import collections

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.datasets.sgd_dataset.prediction_utils as pred_utils
from nemo.collections.nlp.data.datasets.sgd_dataset.schema_processor import SchemaPreprocessor
from nemo.collections.nlp.nm.trainables import SGDDecoderNM, SGDEncoderNM
from nemo.core import CheckpointCallback, EvaluatorCallback, NeuralModuleFactory, SimpleLossLoggerCallback
from nemo.utils import logging
from nemo.utils.lr_policies import get_lr_policy

import nemo.collections.nlp.data.datasets.sgd_dataset.metrics as metrics
from nemo.utils import logging

__all__ = [
    'get_in_domain_services',
    'get_dataset_as_dict',
    'ALL_SERVICES',
    'SEEN_SERVICES',
    'UNSEEN_SERVICES',
    'get_metrics',
    'PER_FRAME_OUTPUT_FILENAME',
]

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

# Name of the file containing all predictions and their corresponding frame metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path) as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
        f.close()
    return service_set


def get_in_domain_services(schema_path, service_set):
    """Get the set of common services between a schema and set of services.
    Args:
        schema_path (str): path to schema file
        service_set (set): set of services
    """
    return get_service_set(schema_path) & service_set


def get_dataset_as_dict(file_path_patterns):
    """Read the DSTC8/SGD json dialogue data as dictionary with dialog ID as keys."""
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp:
            continue
        logging.debug("Loading file: %s", fp)
        with open(fp) as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dataset_dict[dial["dialogue_id"]] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
            f.close()
    return dataset_dict


def get_metrics(dataset_ref, dataset_hyp, service_schemas, in_domain_services, joint_acc_across_turn, no_fuzzy_match):
    """Calculate the DSTC8/SGD metrics.

  Args:
    dataset_ref: The ground truth dataset represented as a dict mapping dialogue
      id to the corresponding dialogue.
    dataset_hyp: The predictions in the same format as `dataset_ref`.
    service_schemas: A dict mapping service name to the schema for the service.
    in_domain_services: The set of services which are present in the training
      set.
    schemas: Schemas with information for all services

  Returns:
    A dict mapping a metric collection name to a dict containing the values
    for various metrics. Each metric collection aggregates the metrics across
    a specific set of frames in the dialogues.
  """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.
    metric_collections = collections.defaultdict(lambda: collections.defaultdict(list))

    # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
    assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
    logging.debug("len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp), len(dataset_ref))

    # Store metrics for every frame for debugging.
    per_frame_metric = {}

    for dial_id, dial_hyp in dataset_hyp.items():
        seen = True
        dial_ref = dataset_ref[dial_id]

        if set(dial_ref["services"]) != set(dial_hyp["services"]):
            raise ValueError(
                "Set of services present in ground truth and predictions don't match "
                "for dialogue with id {}".format(dial_id)
            )

        joint_metrics = [metrics.JOINT_GOAL_ACCURACY, metrics.JOINT_CAT_ACCURACY, metrics.JOINT_NONCAT_ACCURACY]
        for turn_id, (turn_ref, turn_hyp) in enumerate(zip(dial_ref["turns"], dial_hyp["turns"])):
            metric_collections_per_turn = collections.defaultdict(lambda: collections.defaultdict(lambda: 1.0))
            if turn_ref["speaker"] != turn_hyp["speaker"]:
                raise ValueError("Speakers don't match in dialogue with id {}".format(dial_id))

            # Skip system turns because metrics are only computed for user turns.
            if turn_ref["speaker"] != "USER":
                continue

            if turn_ref["utterance"] != turn_hyp["utterance"]:
                logging.error("Ref utt: %s", turn_ref["utterance"])
                logging.error("Hyp utt: %s", turn_hyp["utterance"])
                raise ValueError("Utterances don't match for dialogue with id {}".format(dial_id))

            hyp_frames_by_service = {frame["service"]: frame for frame in turn_hyp["frames"]}

            # Calculate metrics for each frame in each user turn.
            for frame_ref in turn_ref["frames"]:
                service_name = frame_ref["service"]
                if service_name not in hyp_frames_by_service:
                    raise ValueError(
                        "Frame for service {} not found in dialogue with id {}".format(service_name, dial_id)
                    )
                service = service_schemas[service_name]
                frame_hyp = hyp_frames_by_service[service_name]

                active_intent_acc = metrics.get_active_intent_accuracy(frame_ref, frame_hyp)
                slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
                    frame_ref, frame_hyp, turn_ref["utterance"], service
                )
                requested_slots_f1_scores = metrics.get_requested_slots_f1(frame_ref, frame_hyp)
                goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
                    frame_ref, frame_hyp, service, no_fuzzy_match
                )

                frame_metric = {
                    metrics.ACTIVE_INTENT_ACCURACY: active_intent_acc,
                    metrics.REQUESTED_SLOTS_F1: requested_slots_f1_scores.f1,
                    metrics.REQUESTED_SLOTS_PRECISION: requested_slots_f1_scores.precision,
                    metrics.REQUESTED_SLOTS_RECALL: requested_slots_f1_scores.recall,
                }
                if slot_tagging_f1_scores is not None:
                    frame_metric[metrics.SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
                    frame_metric[metrics.SLOT_TAGGING_PRECISION] = slot_tagging_f1_scores.precision
                    frame_metric[metrics.SLOT_TAGGING_RECALL] = slot_tagging_f1_scores.recall
                frame_metric.update(goal_accuracy_dict)

                frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id, frame_hyp["service"])
                per_frame_metric[frame_id] = frame_metric
                # Add the frame-level metric result back to dialogues.
                frame_hyp["metrics"] = frame_metric

                # Get the domain name of the service.
                domain_name = frame_hyp["service"].split("_")[0]
                domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
                if frame_hyp["service"] in in_domain_services and seen:
                    domain_keys.append(SEEN_SERVICES)
                else:
                    seen = False
                    domain_keys.append(UNSEEN_SERVICES)
                for domain_key in domain_keys:
                    for metric_key, metric_value in frame_metric.items():
                        if metric_value != metrics.NAN_VAL:
                            if joint_acc_across_turn and metric_key in joint_metrics:
                                metric_collections_per_turn[domain_key][metric_key] *= metric_value
                            else:
                                metric_collections[domain_key][metric_key].append(metric_value)
            if joint_acc_across_turn:
                # Conduct multiwoz style evaluation that computes joint goal accuracy
                # across all the slot values of all the domains for each turn.
                for domain_key in metric_collections_per_turn:
                    for metric_key, metric_value in metric_collections_per_turn[domain_key].items():
                        metric_collections[domain_key][metric_key].append(metric_value)

    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = round(float(np.mean(value_list)) * 100.0, 2)
            else:
                domain_metric_aggregate[metric_key] = metrics.NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    return all_metric_aggregate, per_frame_metric

__all__ = ['InputExample', 'STR_DONTCARE', 'STATUS_OFF', 'STATUS_ACTIVE', 'STATUS_DONTCARE', 'truncate_seq_pair']

STR_DONTCARE = "dontcare"

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2
STATUS_CARRY = 3

class InputExample(object):
    """An example for training/inference."""

    def __init__(
        self,
        schema_config,
        service_schema,
        example_id="NONE",
        example_id_num=[],
        is_real_example=False,
        tokenizer=None,
    ):
        """Constructs an InputExample.

        Args:
          max_seq_length: The maximum length of the sequence. Sequences longer than
            this value will be truncated.
          service_schema: A ServiceSchema object wrapping the schema for the service
            corresponding to this example.
          example_id: Unique identifier for the example, like: 'train-1_00000-00-Restaurants_1'
          example_id_num: dialogue_id and turn_id combined and service id combined into a list of ints,
            like: [1, 0, 0, 18]
          is_real_example: Indicates if an example is real or used for padding in a
            minibatch.
          tokenizer (Tokenizer): such as NemoBertTokenizer
        """
        self.schema_config = schema_config
        self.service_schema = service_schema
        self.example_id = example_id
        self.example_id_num = example_id_num
        self._add_carry_value = service_schema._add_carry_value
        self._add_carry_status = service_schema._add_carry_status

        self.is_real_example = is_real_example
        self._max_seq_length = schema_config["MAX_SEQ_LENGTH"]
        self._tokenizer = tokenizer
        if self.is_real_example and self._tokenizer is None:
            raise ValueError("Must specify tokenizer when input is a real example.")

        self.user_utterance = ''
        self.system_utterance = ''
        # The id of each subword in the vocabulary for BERT.
        self.utterance_ids = [0] * self._max_seq_length
        # Denotes the identity of the sequence. Takes values 0 (system utterance) and 1 (user utterance).
        self.utterance_segment = [0] * self._max_seq_length
        # Mask which takes the value 0 for padded tokens and 1 otherwise.
        self.utterance_mask = [0] * self._max_seq_length
        # Start and inclusive end character indices in the original utterance
        # corresponding to the tokens. This is used to obtain the character indices
        # from the predicted subword indices during inference.
        # NOTE: A positive value indicates the character indices in the user
        # utterance whereas a negative value indicates the character indices in the
        # system utterance. The indices are offset by 1 to prevent ambiguity in the
        # 0 index, which could be in either the user or system utterance by the
        # above convention. Now the 0 index corresponds to padded tokens.
        self.start_char_idx = [0] * self._max_seq_length
        self.end_char_idx = [0] * self._max_seq_length

        # Number of categorical slots present in the service.
        self.num_categorical_slots = 0
        # The status of each categorical slot in the service.
        self.categorical_slot_status = [STATUS_OFF] * schema_config["MAX_NUM_CAT_SLOT"]
        # Masks out categorical status for padded cat slots
        self.cat_slot_status_mask = [0] * len(self.categorical_slot_status)
        # Number of values taken by each categorical slot.
        self.num_categorical_slot_values = [0] * schema_config["MAX_NUM_CAT_SLOT"]
        # The index of the correct value for each categorical slot.
        self.categorical_slot_values = [0] * schema_config["MAX_NUM_CAT_SLOT"]
        # Masks out categorical slots values for slots not used in the service
        self.cat_slot_values_mask = [
            [0] * schema_config["MAX_NUM_VALUE_PER_CAT_SLOT"] for _ in range(schema_config["MAX_NUM_CAT_SLOT"])
        ]

        # Number of non-categorical slots present in the service.
        self.num_noncategorical_slots = 0
        # The status of each non-categorical slot in the service.
        self.noncategorical_slot_status = [STATUS_OFF] * schema_config["MAX_NUM_NONCAT_SLOT"]
        # Masks out non-categorical status for padded cat slots
        self.noncat_slot_status_mask = [0] * len(self.noncategorical_slot_status)
        # The index of the starting subword corresponding to the slot span for a
        # non-categorical slot value.
        self.noncategorical_slot_value_start = [0] * schema_config["MAX_NUM_NONCAT_SLOT"]
        # The index of the ending (inclusive) subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_end = [0] * schema_config["MAX_NUM_NONCAT_SLOT"]

        # Total number of slots present in the service. All slots are included here
        # since every slot can be requested.
        self.num_slots = 0
        # Takes value 1 if the corresponding slot is requested, 0 otherwise.
        self.requested_slot_status = [STATUS_OFF] * (
            schema_config["MAX_NUM_CAT_SLOT"] + schema_config["MAX_NUM_NONCAT_SLOT"]
        )
        # Masks out requested slots that are not used for the service
        self.requested_slot_mask = [0] * len(self.requested_slot_status)

        # Total number of intents present in the service.
        self.num_intents = 0
        # Takes value 1 if the intent is active, 0 otherwise.
        self.intent_status = [STATUS_OFF] * schema_config["MAX_NUM_INTENT"]
        # Masks out intents that are not used for the service, [1] for none intent
        self.intent_status_mask = [1] + [0] * len(self.intent_status)
        # Label for active intent in the turn
        self.intent_status_labels = 0

    @property
    def readable_summary(self):
        """Get a readable dict that summarizes the attributes of an InputExample."""
        seq_length = sum(self.utterance_mask)
        utt_toks = self._tokenizer.ids_to_tokens(self.utterance_ids[:seq_length])
        utt_tok_mask_pairs = list(zip(utt_toks, self.utterance_segment[:seq_length]))
        active_intents = [
            self.service_schema.get_intent_from_id(idx)
            for idx, s in enumerate(self.intent_status)
            if s == STATUS_ACTIVE
        ]
        if len(active_intents) > 1:
            raise ValueError("Should not have multiple active intents in a single service.")
        active_intent = active_intents[0] if active_intents else ""
        slot_values_in_state = {}
        for idx, s in enumerate(self.categorical_slot_status):
            if s == STATUS_ACTIVE:
                value_id = self.categorical_slot_values[idx]
                slot_values_in_state[
                    self.service_schema.get_categorical_slot_from_id(idx)
                ] = self.service_schema.get_categorical_slot_value_from_id(idx, value_id)
            elif s == STATUS_DONTCARE:
                slot_values_in_state[self.service_schema.get_categorical_slot_from_id(idx)] = STR_DONTCARE
        for idx, s in enumerate(self.noncategorical_slot_status):
            if s == STATUS_ACTIVE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                start_id = self.noncategorical_slot_value_start[idx]
                end_id = self.noncategorical_slot_value_end[idx]
                # Token list is consisted of the subwords that may start with "##". We
                # remove "##" to reconstruct the original value. Note that it's not a
                # strict restoration of the original string. It's primarily used for
                # debugging.
                # ex. ["san", "j", "##ose"] --> "san jose"
                readable_value = " ".join(utt_toks[start_id : end_id + 1]).replace(" ##", "")
                slot_values_in_state[slot] = readable_value
            elif s == STATUS_DONTCARE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                slot_values_in_state[slot] = STR_DONTCARE

        summary_dict = {
            "utt_tok_mask_pairs": utt_tok_mask_pairs,
            "utt_len": seq_length,
            "num_categorical_slots": self.num_categorical_slots,
            "num_categorical_slot_values": self.num_categorical_slot_values,
            "num_noncategorical_slots": self.num_noncategorical_slots,
            "service_name": self.service_schema.service_name,
            "active_intent": active_intent,
            "slot_values_in_state": slot_values_in_state,
        }
        return summary_dict

    def add_utterance_features(
        self, system_tokens, system_inv_alignments, user_tokens, user_inv_alignments, system_utterance, user_utterance
    ):
        """Add utterance related features input to bert.

        Note: this method modifies the system tokens and user_tokens in place to
        make their total length <= the maximum input length for BERT model.

        Args:
          system_tokens: a list of strings which represents system utterance.
          system_inv_alignments: a list of tuples which denotes the start and end
            charater of the tpken that a bert token originates from in the original
            system utterance.
          user_tokens: a list of strings which represents user utterance.
          user_inv_alignments: a list of tuples which denotes the start and end
            charater of the token that a bert token originates from in the original
            user utterance.
        """
        # Make user-system utterance input (in BERT format)
        # Input sequence length for utterance BERT encoder
        max_utt_len = self._max_seq_length

        # Modify lengths of sys & usr utterance so that length of total utt
        # (including cls_token, setp_token, sep_token) is no more than max_utt_len
        is_too_long = truncate_seq_pair(system_tokens, user_tokens, max_utt_len - 3)
        if is_too_long:
            logging.debug(f'Utterance sequence truncated in example id - {self.example_id}.')

        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for system utterance (sequence A) and
        # user utterance (sequence B).
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(system_tokens):
            utt_subword.append(subword)
            utt_seg.append(0)
            utt_mask.append(1)
            st, en = system_inv_alignments[subword_idx]
            start_char_idx.append(-(st + 1))
            end_char_idx.append(-(en + 1))

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(user_tokens):
            utt_subword.append(subword)
            utt_seg.append(1)
            utt_mask.append(1)
            st, en = user_inv_alignments[subword_idx]
            start_char_idx.append(st + 1)
            end_char_idx.append(en + 1)

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(1)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        utterance_ids = self._tokenizer.tokens_to_ids(utt_subword)

        # Zero-pad up to the BERT input sequence length.
        while len(utterance_ids) < max_utt_len:
            utterance_ids.append(0)
            utt_seg.append(0)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)
        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

        self.user_utterances = user_utterance
        self.system_utterance = system_utterance

    def make_copy_with_utterance_features(self):
        """Make a copy of the current example with utterance features."""
        new_example = InputExample(
            schema_config=self.schema_config,
            service_schema=self.service_schema,
            example_id=self.example_id,
            example_id_num=self.example_id_num,
            is_real_example=self.is_real_example,
            tokenizer=self._tokenizer,
        )
        new_example.utterance_ids = list(self.utterance_ids)
        new_example.utterance_segment = list(self.utterance_segment)
        new_example.utterance_mask = list(self.utterance_mask)
        new_example.start_char_idx = list(self.start_char_idx)
        new_example.end_char_idx = list(self.end_char_idx)
        new_example.user_utterance = self.user_utterance
        new_example.system_utterance = self.system_utterance
        return new_example

    def add_categorical_slots(self, state_update, agg_sys_state):
        """Add features for categorical slots."""
        categorical_slots = self.service_schema.categorical_slots
        self.num_categorical_slots = len(categorical_slots)
        for slot_idx, slot in enumerate(categorical_slots):
            values = state_update.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            self.num_categorical_slot_values[slot_idx] = len(slot_values)
            # set slot mask to 1, i.e. the slot exists in the service
            self.cat_slot_status_mask[slot_idx] = 1
            # set the number of active slot values for this slots in the service
            for slot_value_idx in range(len(self.service_schema._categorical_slot_values[slot])):
                self.cat_slot_values_mask[slot_idx][slot_value_idx] = 1

            if not values:
                self.categorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                self.categorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                value_id = self.service_schema.get_categorical_slot_value_id(slot, values[0])
                if value_id < 0:
                    logging.warning(
                        f"Categorical value not found: EXAMPLE_ID:{self.example_id}, EXAMPLE_ID_NUM:{self.example_id_num}"
                    )
                    logging.warning(f"SYSTEM: {self.system_utterance} || USER: {self.user_utterance}")
                else:
                    if values[0] not in agg_sys_state.get(slot, []):
                        self.categorical_slot_status[slot_idx] = STATUS_ACTIVE
                        self.categorical_slot_values[slot_idx] = value_id
                    else:
                        if self._add_carry_status:
                            self.categorical_slot_status[slot_idx] = STATUS_CARRY
                        else:
                            self.categorical_slot_status[slot_idx] = STATUS_ACTIVE

                        if self._add_carry_value:
                            self.categorical_slot_values[slot_idx] = self.service_schema.get_categorical_slot_value_id(
                                slot, "#CARRYVALUE#"
                            )
                            logging.debug(
                                f"Found slot:{slot}, value:{values[0]}, slot_id:{self.categorical_slot_values[slot_idx]} in prev states: {agg_sys_state}"
                            )
                        else:
                            self.categorical_slot_values[slot_idx] = value_id

    def add_noncategorical_slots(self, state_update, system_span_boundaries, user_span_boundaries):
        """Add features for non-categorical slots."""
        noncategorical_slots = self.service_schema.non_categorical_slots
        self.num_noncategorical_slots = len(noncategorical_slots)
        for slot_idx, slot in enumerate(noncategorical_slots):
            values = state_update.get(slot, [])
            self.noncat_slot_status_mask[slot_idx] = 1
            if not values:
                self.noncategorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                self.noncategorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                self.noncategorical_slot_status[slot_idx] = STATUS_ACTIVE
                # Add indices of the start and end tokens for the first encountered
                # value. Spans in user utterance are prioritized over the system
                # utterance. If a span is not found, the slot value is ignored.
                if slot in user_span_boundaries:
                    start, end = user_span_boundaries[slot]
                elif slot in system_span_boundaries:
                    start, end = system_span_boundaries[slot]
                else:
                    # A span may not be found because the value was cropped out or because
                    # the value was mentioned earlier in the dialogue. Since this model
                    # only makes use of the last two utterances to predict state updates,
                    # it will fail in such cases.
                    if self._add_carry_status:
                        self.noncategorical_slot_status[slot_idx] = STATUS_CARRY

                    logging.debug(
                        f'"Slot values {str(values)} not found in user or system utterance in example with id - {self.example_id}.'
                    )
                    start = 0
                    end = 0
                self.noncategorical_slot_value_start[slot_idx] = start
                self.noncategorical_slot_value_end[slot_idx] = end

    def add_requested_slots(self, frame):
        all_slots = self.service_schema.slots
        self.num_slots = len(all_slots)
        for slot_idx, slot in enumerate(all_slots):
            self.requested_slot_mask[slot_idx] = 1
            if slot in frame["state"]["requested_slots"]:
                self.requested_slot_status[slot_idx] = STATUS_ACTIVE

    def add_intents(self, frame):
        all_intents = self.service_schema.intents
        self.num_intents = len(all_intents)
        for intent_idx, intent in enumerate(all_intents):
            if intent == frame["state"]["active_intent"]:
                self.intent_status[intent_idx] = STATUS_ACTIVE
                # adding +1 to take none intent into account
                # supports only 1 active intent in the turn
                self.intent_status_labels = intent_idx + 1
            self.intent_status_mask[intent_idx + 1] = 1

# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate a seq pair in place so that their total length <= max_length."""
    is_too_long = False
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        is_too_long = True
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return is_too_long

__all__ = ['FILE_RANGES', 'PER_FRAME_OUTPUT_FILENAME', 'SGDDataProcessor']

FILE_RANGES = {
    "dstc8_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12), "custom": range(1, 2)},
    "dstc8_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "dstc8_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "multiwoz": {"train": range(1, 18), "dev": range(1, 3), "test": range(1, 3)},
    "debug_sample": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2), "custom": range(1, 2)},
    "sgdplus_single": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
    "sgdplus_all": {"train": range(1, 3), "dev": range(1, 3), "test": range(1, 3)},
}

# Name of the file containing all predictions and their corresponding frame metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


class SGDDataProcessor(object):
    """Data generator for SGD dialogues."""

    def __init__(self, task_name, data_dir, dialogues_example_dir, tokenizer, schema_emb_processor, overwrite_dial_files=False,):
        """
        Constructs SGD8DataProcessor
        Args:
            task_name (str): task  name, for  example, "single_domain"
            data_dir (str): path to data directory
            dialogues_example_dir (str): path to  store processed dialogue examples
            tokenizer (Tokenizer): such as NemoBertTokenizer
            schema_emb_processor (Obj): contains information about schemas
            overwrite_dial_files (bool): whether to overwite dialogue files
        """
        self.data_dir = data_dir
        self.dialogues_examples_dir = dialogues_example_dir

        self._task_name = task_name
        self.schema_config = schema_emb_processor.schema_config
        self.schema_emb_processor = schema_emb_processor

        train_file_range = FILE_RANGES[task_name]["train"]
        dev_file_range = FILE_RANGES[task_name]["dev"]
        test_file_range = FILE_RANGES[task_name]["test"]
        custom_file_range = FILE_RANGES[task_name]["custom"]

        self._file_ranges = {
            "train": train_file_range,
            "dev": dev_file_range,
            "test": test_file_range,
            "custom": custom_file_range,
        }

        self._seen_services = {
            "train": set(),
            "dev": set(),
            "test": set(),
            "custom": set()
        }

        self._tokenizer = tokenizer
        self._max_seq_length = self.schema_config["MAX_SEQ_LENGTH"]

        self.dial_files = {}

        # slots_relation_list.np would contain the candidate list of slots for each (service, slot) which would be
        # looked into when a switch between two services happens in the dialogue and we can not find any value for a slot in the current user utterance.
        # This file would get generated from the dialogues in the training set.
        self.slots_relation_file = os.path.join(dialogues_example_dir, f"{task_name}_train_slots_relation_list.np")

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        for dataset in ["train", "dev", "test", "custom"]:
            # Process dialogue files
            dial_file = f"{task_name}_{dataset}_examples.processed"
            dial_file = os.path.join(dialogues_example_dir, dial_file)
            self.dial_files[(task_name, dataset)] = dial_file

            dialog_paths = SGDDataProcessor.get_dialogue_files(data_dir, dataset, task_name)
            dialogs = SGDDataProcessor.load_dialogues(dialog_paths)
            for dialog in dialogs:
                self._seen_services[dataset].update(set(dialog['services']))

            if not os.path.exists(dial_file) or overwrite_dial_files:
                logging.debug(f"Start generating the dialogue examples for {dataset} dataset.")
                if master_device:
                    if not os.path.exists(dialogues_example_dir):
                        os.makedirs(dialogues_example_dir)
                    dial_examples, slots_relation_list = self._generate_dialog_examples(
                        dataset, schema_emb_processor.schemas
                    )
                    with open(dial_file, "wb") as f:
                        np.save(f, dial_examples)

                    if dataset == "train":
                        with open(self.slots_relation_file, "wb") as f:
                            pickle.dump(slots_relation_list, f)
                        logging.debug(
                            f"The slot carry-over list for train set is stored at {self.slots_relation_file}"
                        )

                    logging.debug(f"The dialogue examples for {dataset} dataset saved at {dial_file}")
                logging.debug(f"Finish generating the dialogue examples for {dataset} dataset.")

            # wait until the master process writes to the dialogue processed file
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def get_dialog_examples(self, dataset):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
          dataset(str): can be "train", "dev", or "test".
        Returns:
          examples: a list of `InputExample`s.
        """
        if (self._task_name, dataset) not in self.dial_files or not os.path.exists(
            self.dial_files[(self._task_name, dataset)]
        ):
            raise ValueError(
                f"{dataset} dialogue examples were not processed for {self._task_name} task. Re-initialize SGDDataProcessor and add {dataset} dataset to datasets arg."
            )
        dial_file = self.dial_files[(self._task_name, dataset)]
        logging.info(f"Loading dialogue examples from {dial_file}.")

        with open(dial_file, "rb") as f:
            dial_examples = np.load(f, allow_pickle=True)

        if not os.path.exists(self.slots_relation_file):
            raise ValueError(
                f"Slots relation file {self.slots_relation_file} does not exist. It is needed for the carry-over mechanism of state tracker for switches between services."
            )

        with open(self.slots_relation_file, "rb") as f:
            self.schema_emb_processor.update_slots_relation_list(pickle.load(f))
        logging.info(
            f"Loaded the slot relation list for value carry-over between services from {self.slots_relation_file}."
        )

        return dial_examples

    def get_seen_services(self, dataset_split):
        return self._seen_services[dataset_split]

    def _generate_dialog_examples(self, dataset, schemas):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
          dataset(str): can be "train", "dev", or "test".
          schemas(Schema): for all services and all datasets processed by the schema_processor
        Returns:
          examples: a list of `InputExample`s.
        """
        logging.info(f'Creating examples and slot relation list from the dialogues started...')
        dialog_paths = [
            os.path.join(self.data_dir, dataset, "dialogues_{:03d}.json".format(i)) for i in self._file_ranges[dataset]
        ]
        dialogs = SGDDataProcessor.load_dialogues(dialog_paths)

        examples = []
        slot_carryover_candlist = defaultdict(int)
        services_switch_counts = defaultdict(int)

        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logging.info(f'Processed {dialog_idx} dialogues.')
            examples.extend(
                self._create_examples_from_dialog(
                    dialog, schemas, dataset, slot_carryover_candlist, services_switch_counts
                )
            )

        slots_relation_list = defaultdict(list)
        for slots_relation, relation_size in slot_carryover_candlist.items():
            if relation_size > 0:
                switch_counts = (
                    services_switch_counts[(slots_relation[0], slots_relation[2])]
                    + services_switch_counts[(slots_relation[2], slots_relation[0])]
                )
                relation_size = relation_size / switch_counts
                slots_relation_list[(slots_relation[0], slots_relation[1])].append(
                    (slots_relation[2], slots_relation[3], relation_size)
                )
                slots_relation_list[(slots_relation[2], slots_relation[3])].append(
                    (slots_relation[0], slots_relation[1], relation_size)
                )

        return examples, slots_relation_list

    def _create_examples_from_dialog(self, dialog, schemas, dataset, slot_carryover_candlist, services_switch_counts):
        """
        Create examples for every turn in the dialog.
        Args:
            dialog (dict): dialogue example
            schemas(Schema): for all services and all datasets processed by the schema_processor
            dataset(str): can be "train", "dev", or "test".
            slot_carryover_candlist(dict): a dictionary to keep and count the number of carry-over cases between two slots from two different services
        Returns:
            examples: a list of `InputExample`s.
        """
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        agg_sys_states = defaultdict(dict)
        agg_sys_states_prev = defaultdict(dict)
        frame_service_prev = ""
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            if turn["speaker"] == "SYSTEM":
                agg_sys_states_prev = copy.deepcopy(agg_sys_states)
                for frame in turn["frames"]:
                    for action in frame["actions"]:
                        if action["slot"] and len(action["values"]) > 0:
                            agg_sys_states[frame["service"]][action["slot"]] = action["values"]

            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}
                else:
                    system_utterance = ""
                    system_frames = {}

                if len(user_frames) == 2:
                    frames_list_name = list(user_frames.keys())
                    frames_list_val = list(user_frames.values())
                    user_frames_ordered = OrderedDict()

                    if frame_service_prev != "" and frames_list_name[0] != frame_service_prev:
                        user_frames_ordered[frames_list_name[1]] = frames_list_val[1]
                        user_frames_ordered[frames_list_name[0]] = frames_list_val[0]
                        user_frames = user_frames_ordered

                turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
                turn_examples, prev_states = self._create_examples_from_turn(
                    turn_id,
                    system_utterance,
                    user_utterance,
                    system_frames,
                    user_frames,
                    prev_states,
                    schemas,
                    copy.deepcopy(agg_sys_states_prev),
                    slot_carryover_candlist,
                    services_switch_counts,
                )
                examples.extend(turn_examples)
                frame_service_prev = user_frames[list(user_frames.keys())[-1]]["service"]
        return examples

    def _get_state_update(self, current_state, prev_state):
        """
        Updates dialogue state
        Args:
            current_state (dict): dict of slot - slot values pairs for the current dialogue turn
            prev_state (dict): dict of slot - slot values pairs for the previous dialogue turns
        Returns:
            state_update (dict): dict of slot - slot values pairs that very added/updated during the current
                dialogue turn
        """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    def _create_examples_from_turn(
        self,
        turn_id,
        system_utterance,
        user_utterance,
        system_frames,
        user_frames,
        prev_states,
        schemas,
        agg_sys_states,
        slot_carryover_candlist,
        services_switch_counts,
    ):
        """
        Creates an example for each frame in the user turn.
        Args:
            turn_id (int): turn number
            system_utterance (str): last system utterance
            user_utterance (str): lst user utterance
            system_frames (dict): all system utterances and slot - slot value pairs
            user_frames (dict): all user utterances and slot - slot value pairs
            prev_states (dict): slot - slot value pairs from the previous turns
            schemas (obj): carries information about the service from the current turn
            agg_sys_states (dict): the collection of all the slots and values mentioned by the system until the previous turn
            slot_carryover_candlist (dict): a dictionary to keep and aggregate the counts of the relations found between any two slots
            services_switch_counts (dict): a dictionary to keep and aggregate the number of switches between any two services
        Returns:
            examples: a list of `InputExample`s.
            prev_states (dict): updated dialogue state
        """
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)
        states = {}
        base_example = InputExample(
            schema_config=self.schema_config, is_real_example=True, tokenizer=self._tokenizer, service_schema=schemas
        )
        base_example.example_id = turn_id

        _, dialog_id, turn_id_ = turn_id.split('-')
        dialog_id_1, dialog_id_2 = dialog_id.split('_')
        base_example.example_id_num = [int(dialog_id_1), int(dialog_id_2), int(turn_id_)]
        base_example.add_utterance_features(
            system_tokens, system_inv_alignments, user_tokens, user_inv_alignments, system_utterance, user_utterance
        )

        examples = []
        for service, user_frame in user_frames.items():
            # Create an example for this service.
            example = base_example.make_copy_with_utterance_features()

            example.example_id = "{}-{}".format(turn_id, service)
            _, dialog_id, turn_id_ = turn_id.split('-')
            dialog_id_1, dialog_id_2 = dialog_id.split('_')
            example.example_id_num = [
                int(dialog_id_1),
                int(dialog_id_2),
                int(turn_id_),
                schemas.get_service_id(service),
            ]

            example.service_schema = schemas.get_service_schema(service)
            system_frame = system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            state_update = self._get_state_update(state, prev_states.get(service, {}))
            states[service] = state

            # Populate features in the example.
            example.add_categorical_slots(state_update, agg_sys_states[service])
            # The input tokens to bert are in the format [CLS] [S1] [S2] ... [SEP]
            # [U1] [U2] ... [SEP] [PAD] ... [PAD]. For system token indices a bias of
            # 1 is added for the [CLS] token and for user tokens a bias of 2 +
            # len(system_tokens) is added to account for [CLS], system tokens and
            # [SEP].
            user_span_boundaries = self._find_subword_indices(
                state_update, user_utterance, user_frame["slots"], user_alignments, user_tokens, 2 + len(system_tokens)
            )
            if system_frame is not None:
                system_span_boundaries = self._find_subword_indices(
                    state_update, system_utterance, system_frame["slots"], system_alignments, system_tokens, 1
                )
            else:
                system_span_boundaries = {}
            example.add_noncategorical_slots(state_update, user_span_boundaries, system_span_boundaries)
            example.add_requested_slots(user_frame)
            example.add_intents(user_frame)
            examples.append(example)

            if service not in prev_states and int(turn_id_) > 0:
                prev_service = ""
                for prev_s, prev_slot_value_list in prev_states.items():
                    if prev_s != service:
                        prev_service = prev_s
                services_switch_counts[(prev_service, service)] += 1

                if prev_service in states:
                    prev_slot_value_list = states[prev_service]
                else:
                    prev_slot_value_list = prev_states[prev_service]

                cur_slot_value_list = state_update
                for cur_slot, cur_values in cur_slot_value_list.items():
                    for prev_slot, prev_values in prev_slot_value_list.items():
                        if "True" in cur_values or "False" in cur_values:
                            continue
                        if set(cur_values) & set(prev_values):
                            slot_carryover_candlist[(prev_service, prev_slot, service, cur_slot)] += 1.0

        return examples, states

    def _find_subword_indices(self, slot_values, utterance, char_slot_spans, alignments, subwords, bias):
        """Find indices for subwords corresponding to slot values."""
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for slot_span in char_slot_spans:
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"] : slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries

    def _tokenize(self, utterance):
        """Tokenize the utterance using word-piece tokenization used by BERT.

        Args:
          utterance: A string containing the utterance to be tokenized.

        Returns:
          bert_tokens: A list of tokens obtained by word-piece tokenization of the
            utterance.
          alignments: A dict mapping indices of characters corresponding to start
            and end positions of words (not subwords) to corresponding indices in
            bert_tokens list.
          inverse_alignments: A list of size equal to bert_tokens. Each element is a
            tuple containing the index of the starting and inclusive ending
            character of the word corresponding to the subword. This list is used
            during inference to map word-piece indices to spans in the original
            utterance.
        """
        # utterance = tokenization.convert_to_unicode(utterance)

        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = SGDDataProcessor._naive_tokenize(utterance)
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = []
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self._tokenizer.text_to_tokens(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments

    def get_num_dialog_examples(self, dataset):
        """
        Gets the number of dilaog examples in the data split.
        Args:
          dataset: str. can be "train", "dev", or "test".
        Returns:from nemo_nlp.data.datasets.sgd import data_utils
          example_count: int. number of examples in the specified dataset.
        """
        example_count = 0
        dialog_paths = [
            os.path.join(self.data_dir, dataset, "dialogues_{:03d}.json".format(i)) for i in self._file_ranges[dataset]
        ]
        dst_set = SGDDataProcessor.load_dialogues(dialog_paths)
        for dialog in dst_set:
            for turn in dialog["turns"]:
                if turn["speaker"] == "USER":
                    example_count += len(turn["frames"])
        return example_count

    @classmethod
    def _naive_tokenize(cls, s):
        """
        Tokenizes a string, separating words, spaces and punctuations.
        Args:
            s (str): a string
        Returns:
            seq_tok (list): list of words, spaces and punctuations from the s
        """
        # Spaces and punctuation marks are all retained, i.e. direct concatenation
        # of all the tokens in the sequence will be the original string.
        seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
        return seq_tok

    @classmethod
    def load_dialogues(cls, dialog_json_filepaths):
        """
        Obtain the list of all dialogues from specified json files.
        Args:
            dialog_json_filepaths (list): list of json files
        Returns:
            dialogs  (list): the list of all dialogues
        """
        dialogs = []
        for dialog_json_filepath in sorted(dialog_json_filepaths):
            with open(dialog_json_filepath, 'r') as f:
                dialogs.extend(json.load(f))
                f.close()
        return dialogs

    @classmethod
    def get_dialogue_files(cls, data_dir, dataset_split, task_name):
        """
        Obtain the list of all dialogue json files
        Args:
            data_dir (str): path to the data folde
            dataset_split (str): dev, test or train
            task_name (str): SGD task name, see keys of the FILE_RANGES
        Returns:
            dialogs (list): the list of all dialogue json files paths
        """
        return [
            os.path.join(data_dir, dataset_split, 'dialogues_{:03d}.json'.format(fid))
            for fid in FILE_RANGES[task_name][dataset_split]
        ]



global_vars = dict()

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def get_str_example_id(eval_dataset, ids_to_service_names_dict, example_id_num):
    def format_turn_id(ex_id_num):
        dialog_id_1, dialog_id_2, turn_id, service_id = ex_id_num
        return "{}-{}_{:05d}-{:02d}-{}".format(
            eval_dataset, dialog_id_1, dialog_id_2, turn_id, ids_to_service_names_dict[service_id]
        )

    return list(map(format_turn_id, tensor2list(example_id_num)))


def eval_iter_callback(tensors, schema_processor, eval_dataset):
    global global_vars
    if 'predictions' not in global_vars:
        global_vars['predictions'] = []

    output = {}
    for k, v in tensors.items():
        ind = k.find('~~~')
        if ind != -1:
            output[k[:ind]] = torch.cat(v)

    predictions = {}
    ids_to_service_names_dict = schema_processor.get_ids_to_service_names_dict()
    predictions['example_id'] = get_str_example_id(eval_dataset, ids_to_service_names_dict, output['example_id_num'])

    predictions['service_id'] = output['service_id']
    predictions['is_real_example'] = output['is_real_example']

    # Scores are output for each intent.
    # Note that the intent indices are shifted by 1 to account for NONE intent.
    predictions['intent_status'] = torch.argmax(output['logit_intent_status'], -1)

    # Scores are output for each requested slot.
    predictions['req_slot_status'] = torch.nn.Sigmoid()(output['logit_req_slot_status'])

    # For categorical slots, the status of each slot and the predicted value are output.
    cat_slot_status_dist = torch.nn.Softmax(dim=-1)(output['logit_cat_slot_status'])
    cat_slot_value_dist = torch.nn.Softmax(dim=-1)(output['logit_cat_slot_value'])

    predictions['cat_slot_status'] = torch.argmax(output['logit_cat_slot_status'], axis=-1)
    predictions['cat_slot_status_p'] = torch.max(cat_slot_status_dist, axis=-1)[0]
    predictions['cat_slot_value'] = torch.argmax(output['logit_cat_slot_value'], axis=-1)
    predictions['cat_slot_value_p'] = torch.max(cat_slot_value_dist, axis=-1)[0]

    # For non-categorical slots, the status of each slot and the indices for spans are output.
    noncat_slot_status_dist = torch.nn.Softmax(dim=-1)(output['logit_noncat_slot_status'])

    predictions['noncat_slot_status'] = torch.argmax(output['logit_noncat_slot_status'], axis=-1)
    predictions['noncat_slot_status_p'] = torch.max(noncat_slot_status_dist, axis=-1)[0]

    softmax = torch.nn.Softmax(dim=-1)
    start_scores = softmax(output['logit_noncat_slot_start'])
    end_scores = softmax(output['logit_noncat_slot_end'])

    batch_size, max_num_noncat_slots, max_num_tokens = end_scores.size()
    # Find the span with the maximum sum of scores for start and end indices.
    total_scores = torch.unsqueeze(start_scores, axis=3) + torch.unsqueeze(end_scores, axis=2)
    # Mask out scores where start_index > end_index.
    # device = total_scores.device
    start_idx = torch.arange(max_num_tokens, device=total_scores.device).view(1, 1, -1, 1)
    end_idx = torch.arange(max_num_tokens, device=total_scores.device).view(1, 1, 1, -1)
    invalid_index_mask = (start_idx > end_idx).repeat(batch_size, max_num_noncat_slots, 1, 1)
    total_scores = torch.where(
        invalid_index_mask,
        torch.zeros(total_scores.size(), device=total_scores.device, dtype=total_scores.dtype),
        total_scores,
    )
    max_span_index = torch.argmax(total_scores.view(-1, max_num_noncat_slots, max_num_tokens ** 2), axis=-1)
    max_span_p = torch.max(total_scores.view(-1, max_num_noncat_slots, max_num_tokens ** 2), axis=-1)[0]
    predictions['noncat_slot_p'] = max_span_p

    span_start_index = torch.div(max_span_index, max_num_tokens)
    span_end_index = torch.fmod(max_span_index, max_num_tokens)

    predictions['noncat_slot_start'] = span_start_index
    predictions['noncat_slot_end'] = span_end_index

    # Add inverse alignments.
    predictions['noncat_alignment_start'] = output['start_char_idx']
    predictions['noncat_alignment_end'] = output['end_char_idx']

    # added for debugging
    predictions['cat_slot_status_GT'] = output['categorical_slot_status']
    predictions['noncat_slot_status_GT'] = output['noncategorical_slot_status']
    predictions['cat_slot_value_GT'] = output['categorical_slot_values']

    global_vars['predictions'].extend(combine_predictions_in_example(predictions, batch_size))


def combine_predictions_in_example(predictions, batch_size):
    '''
    Combines predicted values to a single example.
    '''
    examples_preds = [{} for _ in range(batch_size)]
    for k, v in predictions.items():
        if k != 'example_id':
            v = torch.chunk(v, batch_size)

        for i in range(batch_size):
            if k == 'example_id':
                examples_preds[i][k] = v[i]
            else:
                examples_preds[i][k] = v[i].view(-1)
    return examples_preds


def eval_epochs_done_callback(
    task_name,
    eval_dataset,
    data_dir,
    prediction_dir,
    tracker_model,
    eval_debug,
    dialogues_processor,
    schema_emb_preprocessor,
    joint_acc_across_turn,
    no_fuzzy_match,
):
    ##############
    # we'll write predictions to file in Dstc8/SGD format during evaluation callback
    prediction_dir = os.path.join(prediction_dir, 'predictions', 'pred_res_{}_{}'.format(eval_dataset, task_name))
    os.makedirs(prediction_dir, exist_ok=True)

    input_json_files = SGDDataProcessor.get_dialogue_files(data_dir, eval_dataset, task_name)
    pred_utils.write_predictions_to_file(
        global_vars['predictions'],
        input_json_files,
        prediction_dir,
        schemas=schema_emb_preprocessor.schemas,
        tracker_model=tracker_model,
        eval_debug=eval_debug,
        in_domain_services=set(),
    )

# Parsing arguments
parser = argparse.ArgumentParser(description='Schema_guided_dst')

# BERT based utterance encoder related arguments
parser.add_argument(
    "--max_seq_length",
    default=80,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate for BERT representations.")
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-cased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to model checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)

# Hyperparameters and optimization related flags.
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    help="The folder containing the checkpoints for the model to continue training",
)
parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
parser.add_argument("--num_epochs", default=80, type=int, help="Total number of training epochs to perform.")

parser.add_argument("--optimizer_kind", default="adam_w", type=str)
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_policy", default="PolynomialDecayAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument(
    "--lr_warmup_proportion",
    default=0.1,
    type=float,
    help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10% of training.",
)
parser.add_argument("--grad_norm_clip", type=float, default=1, help="Gradient clipping")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--num_gpus", default=1, type=int)

# Input and output paths and other flags.
parser.add_argument(
    "--task_name",
    default="dstc8_single_domain",
    type=str,
    choices=FILE_RANGES.keys(),
    help="The name of the task to train.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory for the downloaded SGD data, which contains the dialogue files"
    " and schema files of all datasets (eg train, dev)",
)
parser.add_argument(
    "--work_dir",
    type=str,
    default="output/SGD",
    help="The output directory where the model checkpoints will be written.",
)
parser.add_argument(
    "--schema_embedding_dir",
    type=str,
    default='schema_embedding_dir',
    help="Directory where .npy file for embedding of entities (slots, values, intents) in the dataset_split's schema are stored.",
)
parser.add_argument(
    "--no_overwrite_schema_emb_files",
    action="store_false",
    help="Whether to generate a new file saving the dialogue examples.",
    dest="overwrite_schema_emb_files",
)
parser.add_argument(
    "--joint_acc_across_turn",
    action="store_true",
    help="Whether to compute joint accuracy across turn instead of across service. Should be set to True when conducting multiwoz style evaluation.",
)
parser.add_argument(
    "--no_fuzzy_match",
    action="store_true",
    help="Whether to use fuzzy string matching when comparing non-categorical slot values. Fuzz match should not be used when conducting multiwoz style evaluation.",
)
parser.add_argument(
    "--dialogues_example_dir",
    type=str,
    default="dialogues_example_dir",
    help="Directory where preprocessed SGD dialogues are stored.",
)
parser.add_argument(
    "--no_overwrite_dial_files",
    action="store_false",
    help="Whether to generate a new file saving the dialogue examples.",
    dest="overwrite_dial_files",
)
parser.add_argument("--no_shuffle", action="store_true", help="Whether to shuffle training data")
parser.add_argument("--no_time_to_log_dir", action="store_true", help="whether to add time to work_dir or not")
parser.add_argument(
    "--eval_dataset",
    type=str,
    default="dev_test",
    choices=["dev", "test", "dev_test"],
    help="Dataset splits for evaluation.",
)
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)

parser.add_argument(
    "--loss_log_freq", default=-1, type=int, help="Frequency of logging loss values, '-1' - at the end of the epoch",
)

parser.add_argument(
    "--loss_reduction",
    default='mean',
    type=str,
    help="specifies the reduction to apply to the final loss, choose 'mean' or 'sum'",
)

parser.add_argument(
    "--eval_epoch_freq", default=1, type=int, help="Frequency of evaluation",
)

parser.add_argument(
    "--num_workers",
    default=2,
    type=int,
    help="Number of workers for data loading, -1 means set it automatically to the number of CPU cores",
)

parser.add_argument(
    "--enable_pin_memory", action="store_true", help="Enables the pin_memory feature of Pytroch's DataLoader",
)

parser.add_argument(
    "--tracker_model",
    type=str,
    default='baseline',
    choices=['baseline', 'nemotracker'],
    help="Specifies the state tracker model",
)
parser.add_argument(
    "--schema_emb_init",
    type=str,
    default='baseline',
    choices=['baseline', 'random', 'last_layer_average'],
    help="Specifies how schema embeddings are generated. Baseline uses ['CLS'] token",
)
parser.add_argument(
    "--train_schema_emb", action="store_true", help="Specifies whether schema embeddings are trainables.",
)
parser.add_argument(
    "--no_attention_head",
    dest='add_attention_head',
    action="store_false",
    help="Whether to use attention when computing projections for slot statuses and categorical slot values. Is always disabled for the baseline model tracker. When specified for nemotracker goes back to using linear projection.",
)
parser.add_argument(
    "--debug_mode", action="store_true", help="Enables debug mode with more info on data preprocessing and evaluation",
)

parser.add_argument(
    "--checkpoints_to_keep", default=1, type=int, help="The number of last checkpoints to keep",
)

parser.add_argument(
    "--no_carry_value",
    action="store_false",
    help="Disables adding carry-over value to all categorical slots for nemotracker.",
    dest="add_carry_value",
)

parser.add_argument(
    "--no_carry_status",
    action="store_false",
    help="Disables adding carry-over status to the slots for nemotracker.",
    dest="add_carry_status",
)

args = parser.parse_args()
logging.info(args)

if args.debug_mode:
    logging.setLevel("DEBUG")

if args.task_name == "multiwoz":
    schema_config = {
        "MAX_NUM_CAT_SLOT": 9,
        "MAX_NUM_NONCAT_SLOT": 4,
        "MAX_NUM_VALUE_PER_CAT_SLOT": 47,
        "MAX_NUM_INTENT": 1,
    }
else:
    schema_config = {
        "MAX_NUM_CAT_SLOT": 6,
        "MAX_NUM_NONCAT_SLOT": 12,
        "MAX_NUM_VALUE_PER_CAT_SLOT": 12,
        "MAX_NUM_INTENT": 4,
    }

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

nf = NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    checkpoint_dir=args.checkpoint_dir,
    files_to_copy=[__file__],
    add_time_to_log_dir=not args.no_time_to_log_dir,
)

pretrained_bert_model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
    pretrained_model_name=args.pretrained_model_name,
    config=args.bert_config,
    vocab=args.vocab_file,
    checkpoint=args.bert_checkpoint,
)

schema_config["EMBEDDING_DIMENSION"] = pretrained_bert_model.hidden_size
schema_config["MAX_SEQ_LENGTH"] = args.max_seq_length

tokenizer = nemo_nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
    vocab_file=args.vocab_file,
    do_lower_case=args.do_lower_case,
)

# disabling add_carry_status and add_carry_value for the SGD Baseline model
if args.tracker_model == "baseline":
    add_carry_status = False
    add_carry_value = False
    add_attention_head = False
else:
    add_carry_status = args.add_carry_status
    add_carry_value = args.add_carry_value
    add_attention_head = args.add_attention_head

hidden_size = pretrained_bert_model.hidden_size

# Run SGD preprocessor to generate and store schema embeddings
schema_preprocessor = SchemaPreprocessor(
    data_dir=args.data_dir,
    schema_embedding_dir=args.schema_embedding_dir,
    schema_config=schema_config,
    tokenizer=tokenizer,
    bert_model=pretrained_bert_model,
    overwrite_schema_emb_files=args.overwrite_schema_emb_files,
    bert_ckpt_dir=args.bert_checkpoint,
    nf=nf,
    add_carry_value=add_carry_value,
    add_carry_status=add_carry_status,
    mode=args.schema_emb_init,
    is_trainable=args.train_schema_emb,
)

dialogues_processor = SGDDataProcessor(
    task_name=args.task_name,
    data_dir=args.data_dir,
    dialogues_example_dir=args.dialogues_example_dir,
    tokenizer=tokenizer,
    schema_emb_processor=schema_preprocessor,
    overwrite_dial_files=args.overwrite_dial_files,
)

# define model pipeline
sgd_encoder = SGDEncoderNM(hidden_size=hidden_size, dropout=args.dropout)
sgd_decoder = SGDDecoderNM(
    embedding_dim=hidden_size, schema_emb_processor=schema_preprocessor, add_attention_head=add_attention_head
)
dst_loss = nemo_nlp.nm.losses.SGDDialogueStateLossNM(add_carry_status=add_carry_status, reduction=args.loss_reduction)


def create_pipeline(dataset_split='train'):
    datalayer = nemo_nlp.nm.data_layers.SGDDataLayer(
        dataset_split=dataset_split,
        dialogues_processor=dialogues_processor,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.enable_pin_memory,
    )
    data = datalayer()

    # Encode the utterances using BERT.
    token_embeddings = pretrained_bert_model(
        input_ids=data.utterance_ids, attention_mask=data.utterance_mask, token_type_ids=data.utterance_segment,
    )
    encoded_utterance, token_embeddings = sgd_encoder(hidden_states=token_embeddings)
    (
        logit_intent_status,
        logit_req_slot_status,
        logit_cat_slot_status,
        logit_cat_slot_value,
        logit_noncat_slot_status,
        logit_noncat_slot_start,
        logit_noncat_slot_end,
    ) = sgd_decoder(
        encoded_utterance=encoded_utterance,
        token_embeddings=token_embeddings,
        utterance_mask=data.utterance_mask,
        cat_slot_values_mask=data.cat_slot_values_mask,
        intent_status_mask=data.intent_status_mask,
        service_ids=data.service_id,
    )
    tensors = [
        data.example_id_num,
        data.service_id,
        data.is_real_example,
        data.start_char_idx,
        data.end_char_idx,
        logit_intent_status,
        logit_req_slot_status,
        logit_cat_slot_status,
        logit_cat_slot_value,
        logit_noncat_slot_status,
        logit_noncat_slot_start,
        logit_noncat_slot_end,
        data.intent_status_labels,
        data.requested_slot_status,
        data.categorical_slot_status,
        data.categorical_slot_values,
        data.noncategorical_slot_status,
    ]

    return tensors

dataset = 'custom'
tensors = create_pipeline(dataset_split=dataset)

logging.warning("Eval")

# tensors = nf.infer(tensors=eval_tensors)
# print(tensors)
# eval_callbacks = [get_eval_callback('custom')]
values_dict = nf.infer(tensors=tensors)
eval_iter_callback(values_dict, schema_preprocessor, dataset)
eval_epochs_done_callback(
    args.task_name,
    dataset,
    args.data_dir,
    nf.work_dir,
    args.tracker_model,
    args.debug_mode,
    dialogues_processor,
    schema_preprocessor,
    args.joint_acc_across_turn,
    args.no_fuzzy_match,
)

# nf.train(
#     tensors_to_optimize=train_tensors,
#     callbacks=[train_callback, ckpt_callback] + eval_callbacks,
#     lr_policy=lr_policy_fn,
#     optimizer=args.optimizer_kind,
#     optimization_params={
#         "num_epochs": args.num_epochs,
#         "lr": args.learning_rate,
#         "eps": 1e-6,
#         "weight_decay": args.weight_decay,
#         "grad_norm_clip": args.grad_norm_clip,
#     },
# )
