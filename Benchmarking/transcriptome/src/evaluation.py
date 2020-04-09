rule evaluation:
    input:
        true_syntax = features["similarity"]["true_syntax"],
        true_semantics = features["similarity"]["true_semantics"],
        predicted = expand('similarity/{sample}.csv',
                            sample = features["final"])
    output:
        syntax_heatmap = expand('similarity/plots/{sample}_syntax.png',
                            sample = features["final"]),
        semantics_heatmap = expand('similarity/plots/{sample}_semantics.png',
                            sample = features["final"]),
        scores = expand('similarity/scores/{sample}.txt',
                            sample = features["final"]),
        syntax_diff = expand('similarity/matrices/{sample}_syntax.csv',
                            sample = features["final"]),
        semantics_diff = expand('similarity/matrices/{sample}_semantics.csv',
                            sample = features["final"])
    script:
        "evaluation.R"
