rule true_similarity_syntax:
    input:
        formatted_sequence = features["data"]["formatted_sequence"]
    output:
        syntax = features["similarity"]["true_syntax"]
    script:
        "similarity_true_syntax.R"


rule true_similarity_semantics:
    input:
        formatted_sequence = features["data"]["formatted_sequence"],
        GO_terms = features["data"]["GO_terms"]
    output:
        semantics_MF = features["similarity"]["true_semantics_MF"],
        semantics_BP = features["similarity"]["true_semantics_BP"],
        semantics_CC = features["similarity"]["true_semantics_CC"]
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
