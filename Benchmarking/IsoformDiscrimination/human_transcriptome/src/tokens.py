singularity: "docker://bioconda/bioconda-utils-build-env"

rule BPE_training1:
    input:
        train = features["data"]["train"]
    output:
        conc_UniProt = features["data"]["concatenated_UniProt"]
    conda:
        "R_dependencies.yml"
    script:
        "train_BPE.R"

rule BPE_training2:
    input:
        conc_UniProt = features["data"]["concatenated_UniProt"]
    output:
        BPE_model = features["encoded_sequence"]["BPE_model"]
    conda:
        "R_dependencies.yml"
    script:
        "train_BPE2.R"

rule generate_tokens:
    input:
        sequence = features["data"]["sequence"],
        BPE_model = features["encoded_sequence"]["BPE_model"]
    output:
        model_vocab = features["encoded_sequence"]["model_vocab"],
        words = features["encoded_sequence"]["words"]
    conda:
        "R_dependencies.yml"
    script:
        "sequence_tokenization.R"

rule TF_IDF:
    input:
        words = features["encoded_sequence"]["words"]
    output:
        TF_IDF = features["encoded_sequence"]["TF_IDF"]
    conda:
        "R_dependencies_TF-IDF.yml"
    script:
        "TF-IDF_scores.R"
