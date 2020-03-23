singularity: "docker://bioconda/bioconda-utils-build-env"

rule rPCP_calculation:
    input:
        PEAKS_prots = features["peptidome"]["PEAKS_prots"],
        PEAKS_peps = features["peptidome"]["PEAKS_peps"],
        ref_proteins = features["peptidome"]["ref_proteins"],
        IP_info = features["peptidome"]["IP_info"],
        gencode_annot = features["peptidome"]["gencode_annot"],
        biomart_annot = features["peptidome"]["biomart_annot"],
        gene_tr_prot = features["peptidome"]["gene_tr_prot"]
    output:
        rPCP = features["peptidome"]["rPCP"]
    log:
        "results/logs/rPCP_calculation.log"
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
        UniProt_filtered = features["peptidome"]["UniProt_filtered"],
        gencode_annot = features["peptidome"]["gencode_annot"],
        biomart_annot = features["peptidome"]["biomart_annot"]
    output:
        formatted_proteome = features["peptidome"]["formatted_proteome"]
    log:
        "results/logs/proteome_formatting.log"
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
        "results/logs/train_BPE.log"
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
        "results/logs/train_BPE2.log"
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
        "results/logs/generate_tokens.log"
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
        "results/logs/TF_IDF.log"
    conda:
        "R_dependencies_TF-IDF.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "TF-IDF_scores.R"

rule biophys_props:
    input:
        formatted_proteome = features["peptidome"]["formatted_proteome"]
    output:
        properties = features["peptidome"]["properties"]
    log:
        "results/logs/biophys_props.log"
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "biophys_props.R"
