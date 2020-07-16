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

rule true_similarity_syntax:
    input:
        batch_sequence = features["data"]["sequence_batch"],
        batch_accessions = features["data"]["acc_batch"]
    output:
        syntax = features["similarity"]["true_syntax"]
    conda:
        "R_dependencies.yml"
    script:
        "similarity_true_syntax.R"

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


rule evaluation:
    input:
        true_syntax = features["similarity"]["true_syntax"],
        true_semantics_MF = features["similarity"]["true_semantics_MF"],
        true_semantics_BP = features["similarity"]["true_semantics_BP"],
        true_semantics_CC = features["similarity"]["true_semantics_CC"],
        predicted = expand('postprocessing/similarity_{sample}.csv',
                            sample = features["final"])
    output:
        scores = expand('similarity/scores/{sample}.txt',
                            sample = features["final"])
    conda:
        "R_dependencies.yml"
    script:
        "evaluation.R"


rule downstream:
    input:
        scores = expand('similarity/scores/{sample}.txt',
                            sample = features["final"])
    output:
        final = touch('mytask.done')
    conda:
        "R_dependencies.yml"
    script:
        "downstream.R"
