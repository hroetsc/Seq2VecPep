singularity: "docker://bioconda/bioconda-utils-build-env"

rule seq2vec_skipgrams:
    input:
        words = features["encoded_sequence"]["words"]
    output:
        skip_grams = expand('results/skipgrams_{sample}.txt',
                                sample = features["params"]),
        ids = expand('results/seq2vec_ids_{sample}.csv',
                                sample = features["params"])
    conda:
        "environment_seq2vec.yml"
    script:
        "skip_gram_NN_1.py"

rule seq2vec_training:
    input:
        skip_grams = expand('results/skipgrams_{sample}.txt',
                                sample = features["params"]),
        ids = expand('results/seq2vec_ids_{sample}.csv',
                                sample = features["params"])
    output:
        metrics = expand('results/model_metrics_{sample}.txt',
                                sample = features["params"]),
        weights = expand('results/seq2vec_weights_{sample}.csv',
                                sample = features["params"])
    conda:
        "environment_base.yml"
    script:
        "skip_gram_NN_2.py"

rule model_metrics:
    input:
        metrics = expand('results/model_metrics_{sample}.txt',
                                sample = features["params"])
    output:
        acc = expand('results/model_acc_{sample}.png',
                                sample = features["params"]),
        loss = expand('results/model_loss_{sample}.png',
                                sample = features["params"])
    conda:
        "R_dependencies.yml"
    script:
        "plot_model.R"

rule sequence_repres:
    input:
        sequences = features["data"]["sequence"],
        weights = expand('results/seq2vec_weights_{sample}.csv',
                                sample = features["params"]),
        ids = expand('results/seq2vec_ids_{sample}.csv',
                                sample = features["params"]),
        TF_IDF = features["encoded_sequence"]["TF_IDF"],
        words = features["encoded_sequence"]["words"]
    output:
        sequence_repres = expand('results/sequence_repres_{sample}.csv',
                                sample = features["params"])
    conda:
        "R_dependencies.yml"
    script:
        "sequence_repres.R"
