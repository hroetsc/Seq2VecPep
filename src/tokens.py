singularity: "docker://bioconda/bioconda-utils-build-env"

rule BPE_training1:
    input:
        params = features["params"]
    output:
        conc_UniProt = features["data"]["concatenated_UniProt"]
    log:
        "results/logs/train_BPE.log"
    benchmark:
        "results/benchmarks/BPE_training1.txt"
    conda:
        "R_dependencies.yml"
    script:
        "train_BPE.R"

rule BPE_training2:
    input:
        params = features["params"],
        conc_UniProt = features["data"]["concatenated_UniProt"]
    output:
        BPE_model = features["encoded_sequence"]["BPE_model"]
    log:
        "results/logs/train_BPE2.log"
    benchmark:
        "results/benchmarks/BPE_training2.txt"
    conda:
        "R_dependencies.yml"
    threads:
        config["threads"]
    script:
        "train_BPE2.R"

rule generate_tokens:
    input:
        params = features["params"],
        BPE_model = features["encoded_sequence"]["BPE_model"]
    output:
        model_vocab = features["encoded_sequence"]["model_vocab"],
        words = features["encoded_sequence"]["words"]
    log:
        "results/logs/generate_tokens.log"
    benchmark:
        "results/benchmarks/generate_tokens.txt"
    conda:
        "R_dependencies.yml"
    threads:
        config["threads"]
    script:
        "sequence_tokenization.R"

rule TF_IDF:
    input:
        words = features["encoded_sequence"]["words"]
    output:
        TF_IDF = features["encoded_sequence"]["TF_IDF"]
    log:
        "results/logs/TF_IDF.log"
    benchmark:
        "results/benchmarks/TF_IDF.txt"
    conda:
        "R_dependencies_TF-IDF.yml"
    script:
        "TF-IDF_scores.R"
