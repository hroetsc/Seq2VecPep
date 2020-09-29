
rule training_w5:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp-subs_v5k.txt',
        ids = '/scratch2/hroetsc/Seq2Vec/results/ids_hp-subs_v5k.csv'
    output:
        metrics = '/scratch2/hroetsc/Seq2Vec/results/hp-subs_v5k_model_metrics_w5_d128.txt'
    benchmark:
        "results/benchmarks/seq2vec_2_w5_d128.txt"
    script:
        "skip_gram_NN_2_v4_new.py"
