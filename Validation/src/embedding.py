rule seq2vec:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        seq2vec = features["data"]["seq2vec"],
        seq2vec_TFIDF = features["data"]["seq2vec_TFIDF"],
        seq2vec_SIF = features["data"]["seq2vec_SIF"],
        seq2vec_CCR = features["data"]["seq2vec_CCR"],
        seq2vec_TFIDF_CCR = features["data"]["seq2vec_TFIDF_CCR"],
        seq2vec_SIF_CCR = features["data"]["seq2vec_SIF_CCR"]
    output:
        seq2vec = features["seq2vec"]["seq2vec"],
        seq2vec_TFIDF = features["seq2vec"]["seq2vec_TFIDF"],
        seq2vec_SIF = features["seq2vec"]["seq2vec_SIF"],
        seq2vec_CCR = features["seq2vec"]["seq2vec_CCR"],
        seq2vec_TFIDF_CCR = features["seq2vec"]["seq2vec_TFIDF_CCR"],
        seq2vec_SIF_CCR = features["seq2vec"]["seq2vec_SIF_CCR"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_seq2vec.R"


rule biophys:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        TF_IDF = features["data"]["TF-IDF"],
        ids = features["data"]["indices"],
        Props = features["data"]["PropMatrix"]
    output:
        biophys = features["biophys"]["biophys"],
        biophys_TFIDF = features["biophys"]["biophys_TFIDF"],
        biophys_SIF = features["biophys"]["biophys_SIF"],
        biophys_CCR = features["biophys"]["biophys_CCR"],
        biophys_TFIDF_CCR = features["biophys"]["biophys_TFIDF_CCR"],
        biophys_SIF_CCR = features["biophys"]["biophys_SIF_CCR"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_biophys.R"


rule random:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        TF_IDF = features["data"]["TF-IDF"],
        ids = features["data"]["indices"]
    output:
        random = features["random"]["random"],
        random_TFIDF = features["random"]["random_TFIDF"],
        random_SIF = features["random"]["random_SIF"],
        random_CCR = features["random"]["random_CCR"],
        random_TFIDF_CCR = features["random"]["random_TFIDF_CCR"],
        random_SIF_CCR = features["random"]["random_SIF_CCR"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_random.R"


rule QSO:
    input:
        formatted_sequence = features["data"]["sequence_batch"]
    output:
        embedding_QSO = features["embeddings"]["QSO"]
    conda:
        "R_dependencies.yml"
    script:
        "embedding_QSO-couplingNumbers.R"


rule termfreq:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"]
    output:
        embedding_termfreq = features["embeddings"]["termfreq"]
    conda:
        "R_dependencies_termfreq.yml"
    script:
        "embedding_termFreq.R"


rule CCR:
    input:
        embedding = expand('postprocessing/{sample}.csv',
                    sample = features["CCR"])
    output:
        CCR_emb = expand('postprocessing/{sample}_CCR.csv',
                    sample = features["CCR"])
    conda:
        "R_dependencies.yml"
    script:
        "CCR.R"


#rule embedding_hybrid:
#    input:
#        words = features["data"]["word_batch"],
#        ids = features["data"]["indices"],
#        w3 = features["data"]["weights_w3"],
#        w5 = features["data"]["weights"],
#        sup = features["data"]["hybrid_sup"],
#        Props = features["data"]["PropMatrix"]
#    output:
#        hybrid_w3w5 = features["embeddings"]["hybrid_w3w5"],
#        hybrid_w3w5biophys = features["embeddings"]["hybrid_w3w5biophys"],
#        hybrid_sup = features["embeddings"]["hybrid_sup"]
#    conda:
#        "R_dependencies.yml"
#    script:
#        "embedding_hybridFeatures.R"
