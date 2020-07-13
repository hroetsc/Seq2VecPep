singularity: "docker://bioconda/bioconda-utils-build-env"

rule seq2vec_skipgrams:
    input:
        params = features["params"],
        words = features["encoded_sequence"]["words"]
    output:
        skip_grams = features["embedded_sequence"]["skip_grams"],
        ids = features["embedded_sequence"]["subword_ids"]
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
        touch("model.plot.done")
    conda:
        "R_dependencies.yml"
    script:
        "plot_model.R"

rule sequence_repres:
    input:
        weights = features["embedded_sequence"]["model"],
        ids = features["embedded_sequence"]["subword_ids"],
        params = features["params"],
        TF_IDF = features["encoded_sequence"]["TF_IDF"],
        words = features["encoded_sequence"]["words"]
    output:
        sequence_repres_seq2vec = features["embedded_sequence"]["sequence_repres_seq2vec"],
        sequence_repres_seq2vec_TFIDF = features["embedded_sequence"]["sequence_repres_seq2vec_TFIDF"],
        sequence_repres_seq2vec_SIF = features["embedded_sequence"]["sequence_repres_seq2vec_SIF"],
        sequence_repres_seq2vec_CCR = features["embedded_sequence"]["sequence_repres_seq2vec_CCR"],
        sequence_repres_seq2vec_TFIDF_CCR = features["embedded_sequence"]["sequence_repres_seq2vec_TFIDF_CCR"],
        sequence_repres_seq2vec_SIF_CCR = features["embedded_sequence"]["sequence_repres_seq2vec_SIF_CCR"]
    benchmark:
        "results/benchmarks/sequence_repres.txt"
    conda:
        "R_dependencies.yml"
    script:
        "sequence_repres.R"
