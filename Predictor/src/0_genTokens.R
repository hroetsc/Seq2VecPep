### SPLICING PREDICTOR ###
# description: train BPE algorithm on human proteome + DB substrates + DB peptides
# input: human proteome, proteasome DB, eukaryotic proteins
# output: bpe model, table with sequences for seq2vec
#         --> token embeddings are subsequently generated in seq2vec pipeline
# author: HR

library(plyr)
library(dplyr)
library(seqinr)
library(stringr)
library(berryFunctions)
library(tokenizers.bpe)

vocab_size = 5e03
threads = 2

### INPUT ###
eukarya = read.fasta("../files/SwissProt_Eukarya_canonicalAndIsoforms.fasta",
                     seqtype = "AA", whole.header = T)
viruses = read.fasta("../files/SwissProt_Viruses_canonicalAndIsoforms.fasta",
                     seqtype = "AA", whole.header = T)
hp = read.csv("../files/proteome_human.csv", stringsAsFactors = F)
DB = read.csv("data/ProteasomeDB.csv", stringsAsFactors = F)


### MAIN PART ###
# merge all protein sequences
print(paste0("number of eukaryotic proteins in BPE training: ", length(eukarya)))
print(paste0("number of viral proteins in BPE training: ", length(viruses)))

merge = function(fasta = "") {
  seqs = rep(NA, length(fasta))
  pb = txtProgressBar(min = 0, max = length(fasta), style = 3)
  
  for (i in 1:length(fasta)) {
    setTxtProgressBar(pb, i)
    seqs[i] = paste(fasta[[i]], sep = "", collapse = "")
  }
  
  return(seqs %>% paste(sep = "", collapse = ""))
}

eukarya.seqs = merge(eukarya)
viruses.seqs = merge(viruses)
seqs = paste(eukarya.seqs,
             viruses.seqs,
             collapse = "", sep = "")
write.table(seqs, "data/concatenated_seqs.txt", sep = "\t",
            row.names = T, col.names = T)

# train BPE algorithm on eukaryotic and viral proteins
bpeModel = bpe("data/concatenated_seqs.txt",
               coverage = .999,
               vocab_size = vocab_size,
               threads = threads,
               model_path = "data/BPE_model.bpe")

# apply on human proteome and DB substrates
SUBS = DB[, c("substrateID", "substrateSeq")] %>% unique()
names(SUBS) = names(hp)
master = rbind(hp, SUBS)

bpeModel = bpe_load_model("data/BPE_model.bpe",
                          threads = threads)

ModelVocab = bpeModel$vocabulary
ModelVocab = tibble::as_tibble(ModelVocab)


sequences.Encoded.list = list()
pb = txtProgressBar(min = 0, max = nrow(master), style = 3)

for(n in 1:nrow(master)){
  setTxtProgressBar(pb, n)
  
  # encode sequence
  PepEncoded = bpe_encode(model = bpeModel, x = as.character(master$seqs[n]), type = "subwords")
  PepEncoded = unlist(PepEncoded)[-1]
  
  # temporary data frame that contains encoded sequence
  currentPeptide = as_tibble(matrix(ncol = ncol(master)+1, nrow = length(PepEncoded)))
  currentPeptide[, c(1:(ncol(currentPeptide)-1))] = as_tibble(lapply(master[n,], rep, length(PepEncoded)))
  currentPeptide[, ncol(currentPeptide)] = PepEncoded
  
  # add to original data frame
  sequences.Encoded.list[[n]] = currentPeptide
}

sequences.Encoded = plyr::ldply(sequences.Encoded.list, rbind) %>% as.data.frame()
colnames(sequences.Encoded) = c(colnames(master), "segmented_seq")

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
  print(words[sort <=1, ])
  words = words[-which(sort <= 1),]
}


# randomly shuffle sequences to make downstream model training more robust
words = words[sample(nrow(words)), ]

### OUTPUT ###
write.csv(words, "data/words_hp-subs_v5k.csv", row.names = F)

