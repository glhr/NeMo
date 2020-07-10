# Copyright 2020 NVIDIA. All Rights Reserved.
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

import argparse
import json
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


"""
This script faciliates to get EER % based on cosine-smilarity 
for HI-MIA dataset.

Args:
    data_root str: Path to embeddings file and also make sure trails_1m file is also
    placed in this path
    emb : test embedding file path 
    emb_labels : embedding labels file path
"""


def get_labels(manifest):
    lines = open(manifest, 'r').readlines()
    test_list = {}
    for idx, line in enumerate(lines):
        line = line.strip()
        dic = json.loads(line)
        structure = dic['audio_filepath'].split('/')[-3:]
        uniq_name = '@'.join(structure)
        if uniq_name in test_list:
            raise KeyError("uniq name is already present")
        test_list[uniq_name] = idx
    return test_list


def get_acc(trial_file='./myExps/hi-mia/', emb='', manifest=''):
    basename = os.path.dirname(emb)
    X_test = np.load(emb)

    assert len(X_test) == len(open(manifest).readlines())

    test_list = get_labels(manifest)

    tmp_file = open(trial_file, 'r').readlines()
    trail_score = open('trial_score.txt', 'w')

    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []

    # for each of trails in trial file
    for line in tqdm(tmp_file):
        line = line.strip()
        truth, x_speaker, y_speaker = line.split()

        x_speaker = x_speaker.split('/')
        x_speaker = '@'.join(x_speaker)

        y_speaker = y_speaker.split('/')
        y_speaker = '@'.join(y_speaker)

        x_idx = test_list[x_speaker]
        y_idx = test_list[y_speaker]

        X = X_test[x_idx]
        Y = X_test[y_idx]

        if x_speaker not in keys:
            keys.append(x_speaker)
            trial_embs.extend([X])

        if y_speaker not in keys:
            keys.append(y_speaker)
            trial_embs.extend([Y])

        # X=Y
        score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
        score = (score + 1) / 2

        all_scores.append(score)
        trail_score.write(str(score) + "\t" + truth)
        truth = int(truth)
        all_keys.append(truth)

        trail_score.write('\n')

    np.save(basename + '/all_embs_voxceleb.npy', np.asarray(trial_embs))
    np.save(basename + '/all_ids_voxceleb.npy', np.asarray(keys))

    return np.asarray(all_scores), np.asarray(all_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_file", help="directory of embeddings location", type=str, required=True)
    parser.add_argument("--emb", help="test embedding file path", type=str, required=True)
    parser.add_argument("--manifest", help="embedding labels file path", type=str, required=True)

    args = parser.parse_args()
    trial_file, emb, manifest = args.trial_file, args.emb, args.manifest

    y_score, y = get_acc(trial_file=trial_file, emb=emb, manifest=manifest)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print("EER: {:.2f}%".format(eer * 100))