shell.executable("/bin/bash")

singularity: "docker://continuumio/miniconda3:4.5.4"

import pandas as pd
import os, multiprocessing
import yaml
from snakemake import utils
from snakemake.utils import min_version
from snakemake import logging

min_version("5.0")
shell.prefix("set -euo pipefail;")

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
features = yaml.load(open("features.yaml", "r+"), Loader=yaml.FullLoader)

snakefiles = "src/"
include: snakefiles + "tokens.py"
include: snakefiles + "seq2vec.py"

rule all:
    input:
        model = features["embedded_sequence"]["model"],
        acc = "results/metrics/model_acc.png",
        loss = "results/metrics/model_loss.png",
        sequence_repres = features["embedded_sequence"]["sequence_representation"],
