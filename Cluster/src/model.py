#singularity: "docker://bioconda/bioconda-utils-build-env"

rule skipgrams:
    input:
        words = features["words"]
    output:
        skip_grams = features["skip_grams"],
        ids = features["ids"]
    params:
        partition=config["skipgrams"]["partition"],
        nodes=config["skipgrams"]["nodes"],
        tasks=config["skipgrams"]["tasks"],
        time=config["skipgrams"]["time"],
        GPU=config["skipgrams"]["GPU"]
    benchmark:
        "results/benchmarks/seq2vec_1.txt"
    conda:
        "environment_seq2vec.yml"
    script:
        "skip_gram_NN_1.py"

rule training:
    input:
        skip_grams = features["skip_grams"],
        ids=features["ids"]
    output:
        weights = features["weights"],
        model = features["model"],
        metrics = features["model_metrics"]
    params:
        partition=config["training"]["partition"],
        nodes=config["training"]["nodes"],
        tasks=config["training"]["tasks"],
        time=config["training"]["time"],
        GPU=config["training"]["GPU"]
    benchmark:
        "results/benchmarks/seq2vec_2.txt"
    conda:
        "environment_base.yml"
    script:
        "skip_gram_NN_2.py"
