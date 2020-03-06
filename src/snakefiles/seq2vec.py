singularity: "docker://bioconda/bioconda-utils-build-env"

rule seq2vec_skip_grams:
    input:
        words = features["encoded_proteome"]["words"]
    output:
        target = features["embedded_proteome"]["target"],
        context = features["embedded_proteome"]["context"],
        label = features["embedded_proteome"]["label"],
        ids = features["embedded_proteome"]["subword_ids"]
    log:
        "results/logs/seq2vec_1.txt"
    benchmark:
        "results/benchmarks/seq2vec_1.txt"
    conda:
        "python_dependencies.yml"
    script:
        "skip_gram_NN_1.py"

rule seq2vec_model_training:
    input:
        target = features["embedded_proteome"]["target"],
        context = features["embedded_proteome"]["context"],
        label = features["embedded_proteome"]["label"],
        ids = features["embedded_proteome"]["subword_ids"]
    output:
        weights = features["embedded_proteome"]["subword_weights"],
        model = "results/embedded_proteome/model.h5",
        metrics = features["embedded_proteome"]["model_metrics"]
    log:
        "results/logs/seq2vec_2.txt"
    benchmark:
        "results/benchmarks/seq2vec_2.txt"
    conda:
        "python_dependencies.yml"
    script:
        "skip_gram_NN_2.py"

rule plot_model_metrics:
    input:
        metrics = features["embedded_proteome"]["model_metrics"]
    output:
        acc = "results/plots/model_acc.png",
        loss = "results/plots/model_loss.png"
    conda:
        "R_dependencies.yml"
    script:
        "plot_model.R"

rule proteome_repres:
    input:
        weights = features["embedded_proteome"]["subword_weights"],
        ids=features["embedded_proteome"]["subword_ids"],
        formatted_proteome = features["peptidome"]["formatted_proteome"],
        TF_IDF = features["encoded_proteome"]["TF_IDF"]
    output:
        proteome_repres = features["embedded_proteome"]["proteome_representation"]
    log:
        "results/logs/proteome_repres.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "proteome_repres.R"

rule plotting:
    input:
        proteome_repres = features["embedded_proteome"]["proteome_representation"]
    output:
        proteome_props = features["embedded_proteome"]["proteome_properties"],
        p_rPCP = "results/plots/rPCP.png",
        p_rPCP_dens = "results/plots/rPCP_dens.png",
        p_F6 = "results/plots/F6.png",
        p_Z3 = "results/plots/Z3.png",
        p_BLOSUM1 = "results/plots/BLOSUM1.png",
        p_charge = "results/plots/charge.png",
        p_pI = "results/plots/pI.png",
        p_hydrophobicity = "results/plots/Hydrophobicity.png",
        p_H_bonding = "results/plots/H_bonding.png",
        p_Polarity = "results/plots/Polarity.png"
    log:
        "results/logs/plotting.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "plotting.R"
