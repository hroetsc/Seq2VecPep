### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment protein sequences into variable length fragments using byte-pair encoding algorithm
# input:        protein dataset from src_proteins.R, mouse SwissProt sequences
# output:       encoded protein sequences, tokens('words') for seq2vec
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

#tmp !!!
# setwd("Documents/ProtTransEmbedding/Snakemake/")
# proteins = read.csv("data/peptidome/formatted_proteome.csv", stringsAsFactors = F, header = T)
# bpeModel = bpe_load_model("results/encoded_proteome/BPE_model.bpe",
#                           threads = 14)

### INPUT ###
# load protein datasets
proteins = read.csv(snakemake@input[["formatted_proteome"]], stringsAsFactors = F, header = T)
proteins = as.data.frame(proteins)
# load the model
bpeModel = bpe_load_model(snakemake@input[["BPE_model"]],
                          threads = 12)
ModelVocab = bpeModel$vocabulary
ModelVocab = tibble::as_tibble(ModelVocab)

### MAIN PART ###
print("PEPTIDE-PAIR ENCODING")
# peptide pair encoding
progressBar = txtProgressBar(min = 0, max = nrow(proteins), style = 3)
proteins.Encoded.list = list()
for (n in 1:nrow(proteins)) {
  setTxtProgressBar(progressBar, n)
  # encode protein sequence
  PepEncoded = bpe_encode(model = bpeModel, x = as.character(proteins$seqs[n]), type = "subwords")
  PepEncoded = unlist(PepEncoded)[-1]
  # temporary data frame that contains encoded protein
  currentPeptide = as_tibble(matrix(ncol = ncol(proteins)+1, nrow = length(PepEncoded)))
  currentPeptide[, c(1:(ncol(currentPeptide)-1))] = as_tibble(lapply(proteins[n,], rep, length(PepEncoded)))
  currentPeptide[, ncol(currentPeptide)] = PepEncoded
  # add to original data frame
  proteins.Encoded.list[[n]] = currentPeptide
}
proteins.Encoded = as.data.frame(ldply(proteins.Encoded.list, rbind))
colnames(proteins.Encoded) = c(colnames(proteins), "segmented_seq")

print("FORMAT OUTPUT")
# evaluation
# keep only words longer than 1 amino acid
#print("only keeping tokens that are longer than 1 amino acid")
proteins.Encoded = na.omit(proteins.Encoded)
#proteins.Encoded = proteins.Encoded[-which(nchar(proteins.Encoded$segmented_seq) == 1),]

# format words: table with UniProtID and corresponding tokens separated by space
proteins.Encoded.split = split.data.frame(proteins.Encoded, proteins.Encoded$UniProtID)
words = matrix(ncol = 2, nrow = length(proteins.Encoded.split))
for (i in 1:length(proteins.Encoded.split)) {
  words[i, 1] = as.character(proteins.Encoded.split[[i]][1,3])
  words[i, 2] = paste(proteins.Encoded.split[[i]][,9], sep = "", collapse = " ")
}
colnames(words) = c("UniProtID", "tokens")
words = as.data.frame(words)

# keep only proteins that are segmented into more than one token
sort = c()
for (i in 1:nrow(words)) {
  sort = c(sort, ncol(str_split(words$tokens[i], coll(" "), simplify = T)))
}
print(paste0("found ", length(which(sort <= 1)), " of ", nrow(words) ," proteins that consist of only one token and is removing them"))
if (length(which(sort <= 1)) > 0) {
  words = words[-which(sort <= 1),]
}

print("randomize protein order")
# randomly shuffle proteins to make downstream model training more robust
words = words[sample(nrow(words)), ]

### OUTPUT ###
# save model vocabulary
write.csv(ModelVocab, file = unlist(snakemake@output[["model_vocab"]]), row.names = F)
# save words
write.csv(words, file = unlist(snakemake@output[["words"]]), row.names = F)
