rule seq2vec_weighting:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        TF_IDF = features["data"]["TF-IDF"],
        ids = features["data"]["indices"],
        weights = features["data"]["weights"]
    output:
        seq2vec_TFIDF = features["weighting"]["seq2vec_TFIDF"],
        seq2vec_SIF = features["weighting"]["seq2vec_SIF"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_seq2vec.R"


rule biophys_weighting:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        TF_IDF = features["data"]["TF-IDF"],
        ids = features["data"]["indices"],
        weights = features["data"]["weights"]
    output:
        biophys_TFIDF = features["weighting"]["biophys_TFIDF"],
        biophys_SIF = features["weighting"]["biophys_SIF"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_biophys.R"


rule random_weighting:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        TF_IDF = features["data"]["TF-IDF"],
        ids = features["data"]["indices"],
        weights = features["data"]["weights"]
    output:
        random_TFIDF = features["weighting"]["random_TFIDF"],
        random_SIF = features["weighting"]["random_SIF"]
    conda:
        "R_dependencies.yml"
    script:
        "weighting_random.R"


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
