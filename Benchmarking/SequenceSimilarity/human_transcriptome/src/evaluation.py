rule evaluation:
    input:
        true_syntax = features["similarity"]["true_syntax"],
        true_semantics_MF = features["similarity"]["true_semantics_MF"],
        true_semantics_BP = features["similarity"]["true_semantics_BP"],
        true_semantics_CC = features["similarity"]["true_semantics_CC"],
        predicted = expand('similarity/{sample}.csv',
                            sample = features["final"])
    output:
        syntax_heatmap = expand('similarity/plots/{sample}_syntax.png',
                            sample = features["final"]),
        semantics_heatmap_MF = expand('similarity/plots/{sample}_semantics_MF.png',
                            sample = features["final"]),
        semantics_heatmap_BP = expand('similarity/plots/{sample}_semantics_BP.png',
                            sample = features["final"]),
        semantics_heatmap_CC = expand('similarity/plots/{sample}_semantics_CC.png',
                            sample = features["final"]),
        scores = expand('similarity/scores/{sample}.txt',
                            sample = features["final"]),
        syntax_diff = expand('similarity/matrices/{sample}_syntax.csv',
                            sample = features["final"]),
        semantics_diff_MF = expand('similarity/matrices/{sample}_semantics_MF.csv',
                            sample = features["final"]),
        semantics_diff_BP = expand('similarity/matrices/{sample}_semantics_BP.csv',
                            sample = features["final"]),
        semantics_diff_CC = expand('similarity/matrices/{sample}_semantics_CC.csv',
                            sample = features["final"])
    script:
        "evaluation.R"
