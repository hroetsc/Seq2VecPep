singularity: "docker://bioconda/bioconda-utils-build-env"

rule seq2vec_skip_grams:
    input:
        words = features["encoded_proteome"]["words"]
    output:
        skip_grams = features["embedded_proteome"]["skip_grams"],
        ids = features["embedded_proteome"]["subword_ids"]
    log:
        "results/logs/seq2vec_1.log"
    benchmark:
        "results/benchmarks/seq2vec_1.txt"
    conda:
        "python_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "skip_gram_NN_1.py"

rule seq2vec_model_training:
    input:
        skip_grams = features["embedded_proteome"]["skip_grams"],
        ids=features["embedded_proteome"]["subword_ids"]
    output:
        weights = features["embedded_proteome"]["subword_weights"],
        model = "results/embedded_proteome/model.h5",
        metrics = features["embedded_proteome"]["model_metrics"]
    log:
        "results/logs/seq2vec_2.log"
    benchmark:
        "results/benchmarks/seq2vec_2.txt"
    conda:
        "python_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "skip_gram_NN_2.py"

rule plot_model_metrics:
    input:
        metrics = features["embedded_proteome"]["model_metrics"]
    output:
        acc = "results/metrics/model_acc.png",
        loss = "results/metrics/model_loss.png"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "plot_model.R"

rule proteome_repres: # with TF_IDF
    input:
        weights = features["embedded_proteome"]["subword_weights"],
        ids=features["embedded_proteome"]["subword_ids"],
        formatted_proteome = features["peptidome"]["formatted_proteome"],
        TF_IDF = features["encoded_proteome"]["TF_IDF"],
        words = features["encoded_proteome"]["words"]
    output:
        proteome_repres = features["embedded_proteome"]["proteome_representation"],
        proteome_repres_random = features["embedded_proteome"]["proteome_representation_random"],
        KS = features["embedded_proteome"]["KS"]
    log:
        "results/logs/proteome_repres.log"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "proteome_repres.R"

rule plotting:
    input:
        proteome_repres = features["embedded_proteome"]["proteome_representation"],
        proteome_repres_random = features["embedded_proteome"]["proteome_representation_random"],
        properties = features["peptidome"]["properties"]
    output:
        proteome_props = features["embedded_proteome"]["proteome_properties"],
        proteome_props_random = features["embedded_proteome"]["proteome_properties_random"],
        #plot = "results/plots/"
    log:
        "results/logs/plotting.log"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "plotting.R"
