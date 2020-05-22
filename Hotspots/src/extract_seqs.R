### HEADER ###
# HOTSPOT REGIONS
# description: extract hotspot regions and sample N hotspot / non-hotspot sequences
# input: hotspot positions (Juliane), human proteome sequences, tokenized proteome (words)
# output: N random hotspot / non-hotspot regions (tokens)
#         tokens: minimal and extended version
# author: HR


library(plyr)
library(dplyr)
library(stringr)



### INPUT ###
# accessions
load("HOTSPOTS/accU.RData")
# hotspot positions
load("HOTSPOTS/allHspot.RData")
# human proteome
prots = read.csv("../files/proteome_human.csv", stringsAsFactors = F, header = T)
# words
words = read.csv("../RUNS/HumanProteome/words_hp.csv", stringsAsFactors = F, header = T)


### MAIN PART ###
# keep only proteins with hotspots
prots = prots[which(prots$Accession %in% accU), ]
prots = left_join(prots, words)

# flatten list --> data frame
names(allHspot) = accU
pos = plyr::ldply(allHspot)
colnames(pos) = c("Accession", "start", "end")

# hotspot sequences
h.spot = left_join(pos, prots) %>% na.omit()
h.spot[, "hotspot"] = str_sub(h.spot$seqs, start = h.spot$start, end = h.spot$end)


# get non-hotspot sequences
accU = as.character(h.spot$Accession)
nh.spot = list()

pb = txtProgressBar(min = 0, max = length(accU), style = 3)

for (p in 1:length(accU)){
  
  setTxtProgressBar(pb, p)
    
  tmp = h.spot[which(h.spot$Accession %in% accU[p]), ]
  sequ = tmp$seqs[1]
  
  x = strsplit(sequ, "") %>% unlist %>% as.character()
  
  # replace all hotspot substrings by space
  for (i in 1:nrow(tmp)){
    pos_range = seq(tmp$start[i], tmp$end[i])
    sequ = str_replace(sequ, paste(x[pos_range], collapse = ""), " ")
    
  }
  
  # append to list
  nh.spot[[p]] = str_split(sequ, pattern = coll(" "), simplify = T) %>% t()

}

names(nh.spot) = accU
nh.spot = plyr::ldply(nh.spot)
colnames(nh.spot) = c("Accession", "non_hotspot")


# filter non-hotspot regions: must contain at least the same number of aa as minimum of hotspot
# at the same time, remove empty rows
limit = min(nchar(h.spot$hotspot))
nh.spot = nh.spot[-which(nchar(as.character(nh.spot$non_hotspot)) < limit), ]


# add metainformation to n-hsp table
nh.spot = left_join(prots, nh.spot) %>% na.omit()
h.spot$start = NULL
h.spot$end = NULL

h.spot[, "label"] = rep("hotspot", nrow(h.spot))
nh.spot[, "label"] = rep("non_hotspot", nrow(nh.spot))

colnames(nh.spot) = colnames(h.spot)
master = rbind(h.spot, nh.spot)
colnames(master) = c("Accession", "seqs", "tokens", "region", "label")

# retrieve corresponing tokens
master = cbind(master, str_locate(master$seqs, master$region))

# minimal and extended substrings
sub.min = master
sub.ext = master

pb = txtProgressBar(min = 0, max = nrow(master), style = 3)

for (m in 1:nrow(master)){
  
  setTxtProgressBar(pb, m)
    
  tokens = str_split(master$tokens[m], coll(" "), simplify = T) %>% t() %>% as.data.frame()
  tokens$V1 = as.character(tokens$V1)
  tokens[, "start"] = rep(NA, nrow(tokens))
  tokens[, "end"] = tokens$start
  
  # token positions
  end = 0
  
  for (t in 1:nrow(tokens)){
    tokens$start[t] = end + 1
    end = end + nchar(tokens$V1[t])
    tokens$end[t] = end
    
  }
  
  r = seq(master$start[m], master$end[m])
  
  min.sub = tokens[which(tokens$start %in% r & tokens$end %in% r), "V1"]
  ext.sub = tokens[which(tokens$start %in% r | tokens$end %in% r), "V1"]
  
  sub.min[m, "extr_token"] = paste(min.sub, collapse = " ")
  sub.ext[m, "extr_token"] = paste(ext.sub, collapse = " ")
  
}

# clean
cleanup = function(tbl){
  tbl$seqs = NULL
  tbl$tokens = NULL
  
  colnames(tbl) = c("Accession", "region", "label", "start", "end", "tokens")
  
  return(tbl)
}

sub.min = cleanup(sub.min)
sub.min = unique(sub.min)
sub.ext = cleanup(sub.ext)
sub.ext = unique(sub.ext)

### OUTPUT ###
write.csv(sub.min, "RegionSimilarity/data/regions_min_substr.csv", row.names = F)
write.csv(sub.ext, "RegionSimilarity/data/regions_ext_substr.csv", row.names = F)
