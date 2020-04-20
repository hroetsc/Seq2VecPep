### HEADER ###
# ISOFORM DISCRIMINATION ABILITY
# description:  evaluate ability to discriminate between isoforms with different window/embedding sizes
# input:        sequence embeddings
# output:       similarity matrix
# author:       HR


# get all files in this directory
fs = list.files(path = "results", pattern = "similarity_", full.names = T)

scores = matrix(ncol = length(fs), nrow = 100)
scores[1,] = c(fs)

for (f in 1:length(fs)){
  m = read.csv(file = fs[f], stringsAsFactors = F)
  m = as.matrix(m)
  
  if("gene" %in% colnames(m)){
    col = which("gene" %in% colnames(m))
    acc = unique(m[,col])
    for (i in 1:length(acc)){
      k = which(str_split_fixed(m[,col], coll("-"), Inf)[,1] == acc[i])
      
      tmp = m[k,k] %>% as.matrix()
      tmp = tmp[,-col] %>% as.numeric()
      
      # z-transform
      tmp = (tmp - mean(tmp)) / sd(tmp)
      
      scores[c(1+i),f] = mean(tmp)
      
    }
    
  } else {
    col = which("Accession" %in% colnames(m))
    acc = unique(str_split_fixed(m[,col], coll("-"), Inf)[,1])
    for (i in 1:length(acc)){
      k = which(str_split_fixed(m[,col], coll("-"), Inf)[,1] == acc[i])
      
      tmp = m[k,k] %>% as.matrix()
      tmp = tmp[,-col] %>% as.numeric()
      
      # z-transform
      tmp = (tmp - mean(tmp)) / sd(tmp)
      
      scores[c(1+i),f] = mean(tmp)
      
    }
  }
  
}

# plot somehow

write.csv(m, file = unlist(snakemake@output[["scores"]]), row.names = F)