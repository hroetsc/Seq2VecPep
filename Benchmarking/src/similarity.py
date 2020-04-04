rule true_similarity_syntax:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        syntax = features["similarity"]["true_syntax"]
    script:
        "similarity_true_syntax.R"


rule true_similarity_semantics:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        semantics = features["similarity"]["true_semantics"]
    script:
        "similarity_true_semantics.R"


rule similarity:
    input:
        embedding = expand('postprocessing/{sample}.csv',
                            sample = features["final"])
    output:
        similarity = expand('similarity/{sample}.csv',
                            sample = features["final"])
    script:
        "similarity.R"
