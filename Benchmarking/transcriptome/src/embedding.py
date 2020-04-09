rule embedding_seq2vec:
    input:
        formatted_sequence = features["data"]["formatted_sequence"],
        words = features["data"]["words"],
        weights = features["data"]["weights"],
        ids = features["data"]["indices"]
    output:
        embedding_seq2vec = features["embeddings"]["seq2vec"]
    script:
        "embedding_seq2vec.R"


rule embedding_random:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        embedding_random = features["embeddings"]["random"]
    script:
        "embedding_random.R"


rule embedding_termfreq:
    input:
        formatted_sequence = features["data"]["formatted_sequence"],
        words = features["data"]["words"]
    output:
        embedding_termfreq = features["embeddings"]["termfreq"]
    script:
        "embedding_termFreq.R"


rule embedding_autocorrelation:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        embedding_DNADCC = features["embeddings"]["DNADCC"]
    script:
        "embedding_autocorrelation.R"


rule embedding_pseudoDNC:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        embedding_DNAPse = features["embeddings"]["DNAPse"]
    script:
        "embedding_pseudoDNC.R"
