# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training and evaluation"""
import argparse
import logging
import run_lib

def parse_args():
    parser = argparse.ArgumentParser(description="Training and evaluation script for score-based generative models.")
    parser.add_argument('--workdir', type=str, required=True, help="Working directory for checkpoints and logs.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help="Run mode: train or eval.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = run_lib.get_config_from_file(args.config)
    print(f'config:{config}')
    if args.mode == 'train':
        run_lib.train(config, args.workdir)
    elif args.mode == 'eval':
        run_lib.evaluate(config, args.workdir)

