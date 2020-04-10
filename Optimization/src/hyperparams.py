rule BPE_training1:
    input:
        params = "hyperparams.csv"
    output:
        conc_UniProt = "data/concatenated_UniProt.txt"
    conda:
        "R_dependencies.yml"
    script:
        "train_BPE.R"


rule BPE_training2:
    input:
        conc_UniProt = "data/concatenated_UniProt.txt",
        params = "hyperparams.csv"
    output:
        BPE_model = 'data/BPE_model.bpe'
    conda:
        "R_dependencies.yml"
    script:
        "train_BPE2.R"


rule generate_tokens:
    input:
        params = "hyperparams.csv",
        BPE_model = 'data/BPE_model.bpe'
    output:
        model_vocab = 'results/model_vocab.csv',
        words = 'results/words.csv'
    conda:
        "R_dependencies.yml"
    script:
        "sequence_tokenization.R"


rule seq2vec_skip_grams:
    input:
        params = "hyperparams.csv",
        words = 'results/words.csv'
    output:
        skip_grams = 'results/skipgrams.txt',
        ids = 'results/seq2vec_ids.csv'
    conda:
        "environment_seq2vec.yml"
    script:
        "skip_gram_NN_1.py"


rule model_hyperopt:
    input:
        params = "hyperparams.csv",
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
    script:
        "skip_gram_NN_2.py"
