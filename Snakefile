shell.executable("/bin/bash")

singularity: "docker://continuumio/miniconda3:4.5.4"
import pandas as pd
import os, multiprocessing
import yaml
from snakemake.utils import min_version

min_version("5.0")
shell.prefix("set -euo pipefail;")

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
features = yaml.load(open("features.yaml", "r+"), Loader=yaml.FullLoader)

snakefiles = "src/snakefiles/"
#include: snakefiles + "folders.py"
include: snakefiles + "tokens.py"
include: snakefiles + "seq2vec.py"

rule all:
    input:
        model_vocab = features["encoded_proteome"]["model_vocab"],
        acc = "results/plots/model_acc.png",
        loss = "results/plots/model_loss.png",
        proteome_repres = features["embedded_proteome"]["proteome_representation"],
        proteome_props = features["embedded_proteome"]["proteome_properties"]
#        p_rPCP = "results/plots/rPCP.png",
#        p_F6 = "results/plots/F6.png",
#        p_Z3 = "results/plots/Z3.png",
#        p_BLOSUM1 = "results/plots/BLOSUM1.png",
#        p_charge = "results/plots/charge.png",
#        p_pI = "results/plots/pI.png",
#        p_hydrophobicity = "results/plots/Hydrophobicity.png",
#        p_H_bonding = "results/plots/H_bonding.png",
#        p_Polarity = "results/plots/Polarity.png"

### snakemake --dag > dag.dot && dot -Tsvg < dag.dot > dag.svg
### snakemake --filegraph > filegraph.dot && dot -Tsvg < filegraph.dot > filegraph.svg


# tr_quant_st=expand("results/trascriptome_quant/stringtie/{sample}/{sample}_t_data.ctab", sample=SAMPLES)


### snakemake --use-conda --use-singularity -r --verbose
### snakemake --filegraph | dot | display
### snakemake --dag | dot | display
