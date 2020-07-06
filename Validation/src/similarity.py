rule true_similarity_semantics:
    input:
        batch_sequence = features["data"]["sequence_batch"],
        batch_accessions = features["data"]["acc_batch"]
    output:
        semantics_MF = features["similarity"]["true_semantics_MF"],
        semantics_BP = features["similarity"]["true_semantics_BP"],
        semantics_CC = features["similarity"]["true_semantics_CC"]
    conda:
        "R_dependencies.yml"
    script:
        "similarity_true_semantics.R"


rule similarity:
    input:
        embedding = expand('postprocessing/{sample}.csv',
                            sample = features["sim"]),
        batch_accessions = features["data"]["acc_batch"]
    output:
        similarity = expand('postprocessing/similarity_{sample}.csv',
                            sample = features["sim"])
    conda:
        "R_dependencies.yml"
    script:
        "similarity.R"
