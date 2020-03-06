singularity: "docker://bioconda/bioconda-utils-build-env"

rule rPCP_calculation:
    input:
        PEAKS_results = features["peptidome"]["PEAKS_results"],
        ref_proteins = features["peptidome"]["ref_proteins"],
        gencode_annot = features["peptidome"]["GENCODE_annot"],
        biomart_annot = features["peptidome"]["BiomaRt_annot"],
        gene_tr_prot = features["peptidome"]["gene_tr_prot"]
    output:
        rPCP = features["peptidome"]["rPCP"]
    #singularity:
    #    "docker pull quay.io/biocontainers/bioconductor-biomart"
    log:
        "results/logs/rPCP_calculation.txt"
    #conda:
    #    "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "rPCP_calculation.R"

rule proteome_formatting:
    input:
        rPCP = features["peptidome"]["rPCP"],
        UniProt_unfiltered = features["peptidome"]["UniProt_unfiltered"]
    output:
        formatted_proteome = features["peptidome"]["formatted_proteome"]
    log:
        "results/logs/proteome_formatting.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "proteome_formatting.R"

rule BPE_training1:
    input:
        UniProt_unfiltered = features["peptidome"]["UniProt_unfiltered"]
    output:
        conc_UniProt = "data/peptidome/concatenated_UniProt.txt"
    log:
        "results/logs/train_BPE.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "train_BPE.R"

rule BPE_training2:
    input:
        conc_UniProt = "data/peptidome/concatenated_UniProt.txt"
    output:
        BPE_model = features["encoded_proteome"]["BPE_model"]
    log:
        "results/logs/train_BPE2.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "train_BPE2.R"

rule generate_tokens:
    input:
        formatted_proteome = features["peptidome"]["formatted_proteome"],
        BPE_model = features["encoded_proteome"]["BPE_model"]
    output:
        model_vocab = features["encoded_proteome"]["model_vocab"],
        words = features["encoded_proteome"]["words"]
    log:
        "results/logs/generate_tokens.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "protein_segmentation.R"

rule TF_IDF:
    input:
        words = features["encoded_proteome"]["words"]
    output:
        TF_IDF = features["encoded_proteome"]["TF_IDF"]
    log:
        "results/logs/TF_IDF.txt"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "TF-IDF_scores.R"
