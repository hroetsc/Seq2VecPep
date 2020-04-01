### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment sequences into variable length fragments using byte-pair encoding algorithm
# input:        dataset with sequences
# output:       encoded sequences, tokens('words') for seq2vec
# author:       HR

print("### TOKENIZATION OF PROTEOME USING BYTE-PAIR ENCODING ALGORITHM ###")

library(tibble)
library(plyr)
library(dplyr)
library(rlist)
library(stringr)
library(seqinr)
library(berryFunctions)
library(tokenizers.bpe)


### INPUT ###
# load sequence datasets
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)
sequences = read.csv(file = params[which(params$parameter == "Seqinput"), "value"],
                     stringsAsFactors = F, header = T)
sequences = as.data.frame(sequences)

args = commandArgs(T)
workers = args[1]

# load the model
threads = as.numeric(params[which(params$parameter == "threads"), "value"])

bpeModel = bpe_load_model(snakemake@input[["BPE_model"]],
                          threads = threads)
ModelVocab = bpeModel$vocabulary
ModelVocab = tibble::as_tibble(ModelVocab)

### MAIN PART ###
print("PEPTIDE-PAIR ENCODING")
# peptide pair encoding
progressBar = txtProgressBar(min = 0, max = nrow(sequences), style = 3)
sequences.Encoded.list = list()
for (n in 1:nrow(sequences)) {
  setTxtProgressBar(progressBar, n)
  # encode sequence
  PepEncoded = bpe_encode(model = bpeModel, x = as.character(sequences$seqs[n]), type = "subwords")
  PepEncoded = unlist(PepEncoded)[-1]
  # temporary data frame that contains encoded sequence
  currentPeptide = as_tibble(matrix(ncol = ncol(sequences)+1, nrow = length(PepEncoded)))
  currentPeptide[, c(1:(ncol(currentPeptide)-1))] = as_tibble(lapply(sequences[n,], rep, length(PepEncoded)))
  currentPeptide[, ncol(currentPeptide)] = PepEncoded
  # add to original data frame
  sequences.Encoded.list[[n]] = currentPeptide
}
sequences.Encoded = as.data.frame(ldply(sequences.Encoded.list, rbind))
colnames(sequences.Encoded) = c(colnames(sequences), "segmented_seq")

print("FORMAT OUTPUT")
sequences.Encoded = na.omit(sequences.Encoded)

# format words: table with Accession and corresponding tokens separated by space
sequences.Encoded.split = split.data.frame(sequences.Encoded, sequences.Encoded$Accession)
words = matrix(ncol = 2, nrow = length(sequences.Encoded.split))
for (i in 1:length(sequences.Encoded.split)) {
  words[i, 1] = as.character(sequences.Encoded.split[[i]][1, "Accession"])
  words[i, 2] = paste(sequences.Encoded.split[[i]][, "segmented_seq"], sep = "", collapse = " ")
}
colnames(words) = c("Accession", "tokens")
words = as.data.frame(words)

# keep only sequences that are segmented into more than one token
sort = c()
for (i in 1:nrow(words)) {
  sort = c(sort, ncol(str_split(words$tokens[i], coll(" "), simplify = T)))
}
print(paste0("found ", length(which(sort <= 1)), " of ", nrow(words) ," sequences that consist of only one token and is removing them"))
if (length(which(sort <= 1)) > 0) {
  words = words[-which(sort <= 1),]
}

print("randomize sequence order")
# randomly shuffle sequences to make downstream model training more robust
words = words[sample(nrow(words)), ]

### OUTPUT ###
# save model vocabulary
write.csv(ModelVocab, file = unlist(snakemake@output[["model_vocab"]]), row.names = F)
# save words
write.csv(words, file = unlist(snakemake@output[["words"]]), row.names = F)
