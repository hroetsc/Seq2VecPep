
rule training_w5:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_v50k_w5.txt',
        ids = '/scratch2/hroetsc/Seq2Vec/results/ids_hp_v50k_w5.csv'
    output:
        metrics = '/scratch2/hroetsc/Seq2Vec/results/hp_v50k_model_metrics_w5_d128.txt'
    benchmark:
        "results/benchmarks/seq2vec_2_w5_d128.txt"
    script:
        "skip_gram_NN_2_v4_new.py"
