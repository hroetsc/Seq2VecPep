rule skipgrams:
    input:
        words = features["words"],
        params = 'params.csv'
    output:
        skip_grams = expand('/scratch2/hroetsc/Seq2Vec/results/skipgrams_{sample}.txt',
                                sample = features["params"]),
        ids = expand('/scratch2/hroetsc/Seq2Vec/results/ids_{sample}.csv',
                                sample = features["params"])
    benchmark:
        "results/benchmarks/seq2vec_1.txt"
    conda:
        "environment_seq2vec.yml"
    script:
        "skip_gram_NN_1.py"
