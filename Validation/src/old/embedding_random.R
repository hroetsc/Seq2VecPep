### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  random embedding based on initialization of embedding layer
#               (he_uniform)
# input:        sequences
# output:       random sequence embedding
# author:       HR

print("### RANDOM EMBEDDINGS ###")

### INPUT ###
# formatted sequences
sequences = read.csv(snakemake@input[["formatted_sequence"]], stringsAsFactors = F, header = T)

### MAIN PART ###
RAND = matrix(ncol = 102, nrow = nrow(sequences))
RAND[, 1] = sequences$Accession
RAND[, 2] = sequences$seqs

# he_uniform
# fan_in equals input units to weight tensor (roughly 5000)
fan_in = 5000
# calculate limit for uniform distribution
limit = sqrt(6 / fan_in)

# sample from uniform distribution to get "weights"
progressBar = txtProgressBar(min = 0, max = nrow(RAND), style = 3)
for (r in 1:nrow(RAND)){
  setTxtProgressBar(progressBar, r)
  RAND[r, 3:ncol(RAND)] = runif(n = 100, min = -limit, max = limit)
}

RAND = as.data.frame(RAND)
colnames(RAND)[1:2] = c("Accession", "seqs")

### OUTPUT ###
write.csv(x = RAND, file = unlist(snakemake@output[["embedding_random"]]), row.names = F)
