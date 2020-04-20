rule true_similarity_syntax:
    input:
        sequence = features["data"]["sequence"]
    output:
        syntax = features["data"]["true_syntax"]
    script:
        "similarity_true_syntax.R"


rule similarity:
    input:
        sequence_repres = expand('results/sequence_repres_{sample}.csv',
                                sample = features["params"])
    output:
        similarity = expand('results/similarity_{sample}.csv',
                                sample = features["params"])
    script:
        "similarity.R"
