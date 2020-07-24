### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  segment sequences into variable length fragments using byte-pair encoding algorithm
# input:        dataset with sequences (have to contain the columns 'seqs' and 'Accession')
# output:       encoded sequences, tokens('words') for Seq2Vec
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
library(future)


### INPUT ###
# load sequence dataset
params = read.csv(snakemake@input[["params"]], stringsAsFactors = F, header = T)
sequences = read.csv(file = params[which(params$parameter == "Seqinput"), "value"],
                     stringsAsFactors = F, header = T)
sequences = as.data.frame(sequences)

# setwd("Documents/QuantSysBios/ProtTransEmbedding/")
# sequences = read.csv("files/ProteasomeDB.csv", stringsAsFactors = F)

# sequences = read.csv("files/proteome_human.csv", stringsAsFactors = F)
# sequences = sequences[which(sequences$Accession == "H7C241"), ]
# sequences = read.csv("GENCODEml_proteome.csv", stringsAsFactors = F)

# load the model
threads = as.numeric(params[which(params$parameter == "threads"), "value"])

bpeModel = bpe_load_model(snakemake@input[["BPE_model"]],
                          threads = threads)

# threads = future::availableCores()
bpeModel = bpe_load_model("../../Seq2Vec/results/encoded_sequence/BPE_model_hp.bpe")


# store byte-pair encoding vocabulary
ModelVocab = bpeModel$vocabulary
ModelVocab = tibble::as_tibble(ModelVocab)


### MAIN PART ###
print("BYTE-PAIR ENCODING")

sequences.Encoded.list = list()

progressBar = txtProgressBar(min = 0, max = nrow(sequences), style = 3)

for(n in 1:nrow(sequences)){
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

sequences.Encoded = ldply(sequences.Encoded.list, rbind) %>% as.data.frame()
colnames(sequences.Encoded) = c(colnames(sequences), "segmented_seq")


print("CONCATENATE TOKENS")

# format words: table with Accession and corresponding tokens separated by space
words = matrix(ncol = 2, nrow = length(sequences.Encoded.list)) %>% as.data.frame()

for (i in 1:length(sequences.Encoded.list)) {
  # pick accession
  words[i, 1] = as.character(sequences.Encoded.list[[i]][1, 1])
  # pick tokens
  df = as.data.frame(sequences.Encoded.list[[i]])
  words[i, 2] = df[,ncol(sequences.Encoded)] %>% as.vector() %>% paste(collapse = " ", sep = " ")
}
colnames(words) = c("Accession", "tokens")

# keep only sequences that are segmented into more than one token
sort = rep(NA, nrow(words))

for (i in 1:nrow(words)) {
  sort[i] = ncol(str_split(words$tokens[i], coll(" "), simplify = T))
}

print(paste0("found ", length(which(sort <= 1)), " of ", nrow(words) ," sequences that consist of only one token and is removing them"))

if (length(which(sort <= 1)) > 0) {
  words = words[-which(sort <= 1),]
}

print("RANDOMIZE SEQUENCE ORDER")
# randomly shuffle sequences to make downstream model training more robust
words = words[sample(nrow(words)), ]


### OUTPUT ###
# save model vocabulary
write.csv(ModelVocab, file = unlist(snakemake@output[["model_vocab"]]), row.names = F)
# save words
write.csv(words, file = unlist(snakemake@output[["words"]]), row.names = F)


# write.csv(ModelVocab, file = "Seq2Vec/results/encoded_sequence/model_vocab_singleProtein.csv", row.names = F)
# write.csv(words, file = "Seq2Vec/results/encoded_sequence/words_singleProtein.csv", row.names = F)

# write.csv(ModelVocab, file = "../../Seq2Vec/results/encoded_sequence/model_vocab_GENCODEml.csv", row.names = F)
# write.csv(words, file = "../../Seq2Vec/results/encoded_sequence/words_GENCODEml.csv", row.names = F)
