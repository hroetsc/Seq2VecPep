
rule training_w5:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_GENCODEml_w5.txt',
        ids = '/scratch2/hroetsc/Seq2Vec/results/ids_GENCODEml_w5.csv'
    output:
        metrics = '/scratch2/hroetsc/Seq2Vec/results/GENCODEml_model_metrics_w5_d100.txt'
    benchmark:
        "results/benchmarks/seq2vec_2_w5_d100.txt"
    script:
        "skip_gram_NN_2_v4_new.py"
