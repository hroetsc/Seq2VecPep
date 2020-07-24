### HEADER ###
# HOTSPOT REGIONS
# description: sample sequences and submit them to sequence repres script
# input: extracted hotspots/non-hotpots
# output: numeric representation of sampled sequences
# author: HR


### INPUT ###
ext_substr = read.csv("data/regions_ext_substr.csv", stringsAsFactors = F)
min_substr = read.csv("data/regions_min_substr.csv", stringsAsFactors = F)

TFIDF.ext = read.csv("data/ext_substr_TFIDF.csv", stringsAsFactors = F)
TFIDF.min = read.csv("data/min_substr_TFIDF.csv", stringsAsFactors = F)


### MAIN PART ###
N = 1e04
keep = sample(nrow(ext_substr), N)

ext_substr = ext_substr[keep, ]
min_substr = min_substr[keep, ]

# unique Accessions
ext_substr$Accession = paste0(ext_substr$Accession, "_", seq(1, nrow(ext_substr)))
min_substr$Accession = paste0(min_substr$Accession, "_", seq(1, nrow(min_substr)))

# submit to sequence representation
# use weights and IDs from human proteome embedding
{
sequences.master = ext_substr[, c("Accession", "region", "tokens", "label")]
TF_IDF = TFIDF.ext
out = "ext_substr_w5_d100_seq2vec.csv"
out.tfidf = "ext_substr_w5_d100_seq2vec-TFIDF.csv"
out.sif = "ext_substr_w5_d100_seq2vec-SIF.csv"
out.ccr = "ext_substr_w5_d100_seq2vec_CCR.csv"
out.tfidf.ccr = "ext_substr_w5_d100_seq2vec-TFIDF_CCR.csv"
out.sif.ccr = "ext_substr_w5_d100_seq2vec-SIF_CCR.csv"

}


{
  sequences.master = min_substr[, c("Accession", "region", "tokens", "label")]
  TF_IDF = TFIDF.min
  out = "min_substr_w5_d100_seq2vec.csv"
  out.tfidf = "min_substr_w5_d100_seq2vec-TFIDF.csv"
  out.sif = "min_substr_w5_d100_seq2vec-SIF.csv"
  out.ccr = "min_substr_w5_d100_seq2vec_CCR.csv"
  out.tfidf.ccr = "min_substr_w5_d100_seq2vec-TFIDF_CCR.csv"
  out.sif.ccr = "min_substr_w5_d100_seq2vec-SIF_CCR.csv"
  
}

### OUTPUT ###
write.csv(ext_substr, "data/sample_N10k_ext_substr.csv", row.names = F)
write.csv(min_substr, "data/sample_N10k_min_substr.csv", row.names = F)
