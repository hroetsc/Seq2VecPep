
rule data_gen:
    input:
        formatted_sequence = features["data"]["formatted_sequence"],
        words = features["data"]["words"]
    output:
        batch_sequence = features["data"]["sequence_batch"],
        batch_words = features["data"]["word_batch"],
        batch_accessions = features["data"]["acc_batch"]
    conda:
        "R_dependencies.yml"
    script:
        "data_gen.R"
