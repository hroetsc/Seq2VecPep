rule BPE_training:
    input:
        conc_UniProt = "data/concatenated_UniProt.txt"
    output:
        BPE_model = 'data/BPE_model.bpe'
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "train_BPE2.R"

rule generate_tokens:
    input:
        formatted_proteome = 'data/formatted_proteome.csv',
        BPE_model = 'data/BPE_model.bpe'
    output:
        model_vocab = 'results/model_vocab.csv',
        words = 'results/words.csv'
    conda:
        "R_dependencies.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "protein_segmentation.R"

rule seq2vec_skip_grams:
    input:
        words = 'results/words.csv'
    output:
        skip_grams = 'results/skipgrams.txt',
        ids = 'results/seq2vec_ids.csv'
    conda:
        "environment_seq2vec.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "skip_gram_NN_1.py"

rule model_hyperopt:
    input:
        skip_grams = 'results/skipgrams.txt',
        ids = 'results/seq2vec_ids.csv'
    output:
        gp = 'results/hyperopt_gp.csv',
        gbrt = 'results/hyperopt_gbrt.csv',
        gp_conv = 'results/conv_gp.png',
        gbrt_conv = 'results/conv_gbrt.png',
        gp_obj = 'results/obj_gp.png',
        gbrt_obj = 'results/obj_gbrt.png'
    conda:
        "environment_base.yml"
    params:
        n=config["max_cores"],
        mem=config["mem_mb"]
    script:
        "skip_gram_NN_2.py"
