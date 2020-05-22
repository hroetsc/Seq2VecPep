
rule training_w3:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w10.txt',
        ids = '/scratch2/hroetsc/Seq2Vec/results/ids_hp_w10.csv'
    output:
        metrics = '/scratch2/hroetsc/Seq2Vec/results/model_metrics_w10_d100.txt'
    benchmark:
        "results/benchmarks/seq2vec_2_w10_d100.txt"
    script:
        "skip_gram_NN_2_v4.py"
