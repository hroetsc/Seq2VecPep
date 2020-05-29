### HEADER ###
# HOTSPOT REGIONS
# description:  prepare data for classifier
# input:        sequence representation for all proteins
#               extracted hotspot regions
# output:       embedding of sampled regions (ext. and minimal substring)
#               embedding of corresponding protein, label
# author:       HR

library(dplyr)

### INPUT ###
# SIF weighting
prot.emb = read.csv("../RUNS/HumanProteome/results/sequence_repres_seq2vec_CCR.csv",
                    stringsAsFactors = F, header = T)

reg.min = read.csv("RegionSimilarity/data/regions_min_substr.csv",
                   stringsAsFactors = F, header = T)
reg.ext = read.csv("RegionSimilarity/data/regions_ext_substr.csv",
                   stringsAsFactors = F, header = T)

### MAIN PART ###

# same proteins
which(!reg.min$Accession == reg.ext$Accession)

# sample 1000 hostpot and 1000 non-hotspot regions
N = 1000
set.seed(42)
k1 = sample(which(reg.min$label == "hotspot"), N)
k2 = sample(which(reg.min$label == "non_hotspot"), N)

red_and_format = function(tbl = ""){
  
  tbl = tbl[c(k1, k2), ]
  
  return(tbl)
}

reg.min = red_and_format(reg.min)
reg.ext = red_and_format(reg.ext)

which(!reg.min$Accession == reg.ext$Accession)

prot.emb = prot.emb[which(prot.emb$Accession %in% reg.ext$Accession),]

acc = as.data.frame(reg.ext[, c("Accession", "label")])
names(acc) = c("Accession", "label")
prot.str = inner_join(acc, prot.emb)

# --> 3 tables with same proteins/regions in same order

# get embeddings for regions
# run ../src/sequence_repres.R
sequences = reg.ext
sequences.master = reg.ext
sequences = reg.min
sequences.master = reg.min

# do CCR

# read in files
reg.ext = read.csv("repres_ext_regions_w3_d100_seq2vec_CCR.csv",
                   stringsAsFactors = F, header = T)
reg.min = read.csv("repres_min_regions_w3_d100_seq2vec_CCR.csv",
                  stringsAsFactors = F, header = T)

nrow(reg.ext)
nrow(reg.min)

reg.min = reg.min[which(reg.min$Accession %in% prot.str$Accession), ]
reg.ext = reg.ext[which(reg.ext$Accession %in% prot.str$Accession), ]

reg.ext[which(! reg.ext$Accession %in% reg.min$Accession), "Accession"]
prot.str[which(! prot.str$Accession %in% reg.min$Accession), "Accession"]

reg.ext = reg.ext[-which(! reg.ext$Accession %in% reg.min$Accession),]
prot.str = prot.str[-which(! prot.str$Accession %in% reg.min$Accession),]


reg.min = reg.min[order(reg.min$Accession), ]
reg.ext = reg.ext[order(reg.ext$Accession), ]
prot.str = prot.str[order(prot.str$Accession), ]

which(!reg.min$Accession == reg.ext$Accession)

# label vector
labels = c(rep(0, nrow(prot.str)))
labels[which(prot.str$label == "hotspot")] = 1

# shuffle
s = sample(nrow(reg.ext))

reg.ext = reg.ext[s, ]
reg.min = reg.min[s, ]
labels = labels[s]
prot.str = prot.str[s, ]


reg.ext$Accession = NULL
reg.min$Accession = NULL
prot.str$Accession = NULL
prot.str$seqs = NULL
prot.str$tokens = NULL
prot.str$label = NULL


### OUTPUT ###
write.csv(reg.ext, "Classification/ext_substr.csv", row.names = F)
write.csv(reg.min, "Classification/min_substr.csv", row.names = F)
write.csv(prot.str, "Classification/proteins.csv", row.names = F)
write.csv(labels, "Classification/labels.csv", row.names = F)

