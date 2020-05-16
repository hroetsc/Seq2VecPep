
rule evaluation:
    input:
        true_syntax = features["similarity"]["true_syntax"],
        true_semantics_MF = features["similarity"]["true_semantics_MF"],
        true_semantics_BP = features["similarity"]["true_semantics_BP"],
        true_semantics_CC = features["similarity"]["true_semantics_CC"],
        predicted = expand('similarity/{sample}.csv',
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
