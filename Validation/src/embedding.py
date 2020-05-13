rule embedding_seq2vec:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"],
        weights = features["data"]["weights"],
        ids = features["data"]["indices"]
    output:
        embedding_seq2vec = features["embeddings"]["seq2vec"]
    conda:
        "R_dependencies.yml"
    script:
        "embedding_seq2vec.R"


rule embedding_biophys:
    input:
        formatted_sequence = features["data"]["sequence_batch"]
    output:
        embedding_biophys = features["embeddings"]["biophys"]
    conda:
        "R_dependencies.yml"
    script:
        "embedding_biophysProps.R"


rule embedding_QSO:
    input:
        formatted_sequence = features["data"]["sequence_batch"]
    output:
        embedding_QSO = features["embeddings"]["QSO"]
    conda:
        "R_dependencies.yml"
    script:
        "embedding_QSO-couplingNumbers.R"


rule embedding_termfreq:
    input:
        formatted_sequence = features["data"]["sequence_batch"],
        words = features["data"]["word_batch"]
    output:
        embedding_termfreq = features["embeddings"]["termfreq"]
    conda:
        "R_dependencies_termfreq.yml"
    script:
        "embedding_termFreq.R"


rule embedding_random:
    input:
        formatted_sequence = features["data"]["sequence_batch"]
    output:
        embedding_random = features["embeddings"]["random"]
    conda:
        "R_dependencies.yml"
    script:
        "embedding_random.R"
