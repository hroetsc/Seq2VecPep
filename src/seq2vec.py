singularity: "docker://bioconda/bioconda-utils-build-env"

rule seq2vec_skipgrams:
    input:
        params = features["params"],
        words = features["encoded_sequence"]["words"]
    output:
        skip_grams = features["embedded_sequence"]["skip_grams"],
        ids = features["embedded_sequence"]["subword_ids"]
    log:
        "results/logs/seq2vec_1.log"
    benchmark:
        "results/benchmarks/seq2vec_1.txt"
    conda:
        "environment_seq2vec.yml"
    script:
        "skip_gram_NN_1.py"

rule seq2vec_training:
    input:
        params = features["params"],
        skip_grams = features["embedded_sequence"]["skip_grams"],
        ids=features["embedded_sequence"]["subword_ids"]
    output:
        weights = features["embedded_sequence"]["subword_weights"],
        model = features["embedded_sequence"]["model"],
        metrics = features["embedded_sequence"]["model_metrics"]
    log:
        "results/logs/seq2vec_2.log"
    benchmark:
        "results/benchmarks/seq2vec_2.txt"
    conda:
        "environment_base.yml"
    script:
        "skip_gram_NN_2.py"

rule model_metrics:
    input:
        metrics = features["embedded_sequence"]["model_metrics"]
    output:
        acc = "results/metrics/model_acc.png",
        loss = "results/metrics/model_loss.png"
    conda:
        "R_dependencies.yml"
    script:
        "plot_model.R"

rule sequence_repres:
    input:
        weights = features["embedded_sequence"]["subword_weights"],
        ids=features["embedded_sequence"]["subword_ids"],
        params = features["params"],
        TF_IDF = features["encoded_sequence"]["TF_IDF"],
        words = features["encoded_sequence"]["words"]
    output:
        sequence_repres = features["embedded_sequence"]["sequence_representation"],
    log:
        "results/logs/sequence_repres.log"
    benchmark:
        "results/benchmarks/sequence_repres.txt"
    conda:
        "R_dependencies.yml"
    script:
        "sequence_repres.R"
