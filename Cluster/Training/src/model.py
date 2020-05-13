rule training_w1:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w1.txt',
        ids='/scratch2/hroetsc/Seq2Vec/results/ids_hp_w1.csv',
        params = 'params.csv'
    output:
        weights = expand('/scratch2/hroetsc/Seq2Vec/results/weights_{sample}.csv',
                                sample = features["params"]["w1"]),
        model = expand('/scratch2/hroetsc/Seq2Vec/results/model_{sample}.h5',
                                sample = features["params"]["w1"]),
        metrics = expand('/scratch2/hroetsc/Seq2Vec/results/model_metrics_{sample}.txt',
                                sample = features["params"]["w1"]),
    benchmark:
        "results/benchmarks/seq2vec_2_w1.txt"
    script:
        "skip_gram_NN_2_v4.py"


rule training_w3:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w3.txt',
        ids='/scratch2/hroetsc/Seq2Vec/results/ids_hp_w3.csv',
        params = 'params.csv'
    output:
        weights = expand('/scratch2/hroetsc/Seq2Vec/results/weights_{sample}.csv',
                                sample = features["params"]["w3"]),
        model = expand('/scratch2/hroetsc/Seq2Vec/results/model_{sample}.h5',
                                sample = features["params"]["w3"]),
        metrics = expand('/scratch2/hroetsc/Seq2Vec/results/model_metrics_{sample}.txt',
                                sample = features["params"]["w3"]),
    benchmark:
        "results/benchmarks/seq2vec_2_w3.txt"
    script:
        "skip_gram_NN_2_v4.py"


rule training_w5:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w5.txt',
        ids = '/scratch2/hroetsc/Seq2Vec/results/ids_hp_w5.csv'
    output:
        weights = '/scratch2/hroetsc/Seq2Vec/results/weights_w5_d200.csv',
        model = '/scratch2/hroetsc/Seq2Vec/results/model_w5_d200.h5',
        metrics = '/scratch2/hroetsc/Seq2Vec/results/model_metrics_w5_d200.txt'
    benchmark:
        "results/benchmarks/seq2vec_2_w5.txt"
    script:
        "skip_gram_NN_2_v4.py"


rule training_w10:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w10.txt',
        ids='/scratch2/hroetsc/Seq2Vec/results/ids_hp_w10.csv',
        params = 'params.csv'
    output:
        weights = expand('/scratch2/hroetsc/Seq2Vec/results/weights_{sample}.csv',
                                sample = features["params"]["w10"]),
        model = expand('/scratch2/hroetsc/Seq2Vec/results/model_{sample}.h5',
                                sample = features["params"]["w10"]),
        metrics = expand('/scratch2/hroetsc/Seq2Vec/results/model_metrics_{sample}.txt',
                                sample = features["params"]["w10"]),
    benchmark:
        "results/benchmarks/seq2vec_2_w10.txt"
    script:
        "skip_gram_NN_2_v4.py"


rule training_w15:
    input:
        skip_grams = '/scratch2/hroetsc/Seq2Vec/results/skipgrams_hp_w15.txt',
        ids='/scratch2/hroetsc/Seq2Vec/results/ids_hp_w15.csv',
        params = 'params.csv'
    output:
        weights = expand('/scratch2/hroetsc/Seq2Vec/results/weights_{sample}.csv',
                                sample = features["params"]["w15"]),
        model = expand('/scratch2/hroetsc/Seq2Vec/results/model_{sample}.h5',
                                sample = features["params"]["w15"]),
        metrics = expand('/scratch2/hroetsc/Seq2Vec/results/model_metrics_{sample}.txt',
                                sample = features["params"]["w15"]),
    benchmark:
        "results/benchmarks/seq2vec_2_w15.txt"
    script:
        "skip_gram_NN_2_v4.py"
