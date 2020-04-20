### HEADER ###
# ISOFORM DISCRIMINATION ABILITY
# description:  evaluate ability to discriminate between isoforms with different window/embedding sizes
# input:        sequence embeddings
# output:       similarity matrix
# author:       HR


# get all files in this directory
fs = list.files(path = "results/", pattern = "similarity_", full.names = T)

scores = matrix(ncol = length(fs))
scores[1,] = c(fs)

for (f in 1:length(fs)){
  m = read.csv(file = fs[f], stringsAsFactors = F)
  m = as.matrix(m)
  
  if("gene" %in% colnames(m)){
    acc = unique(m$gene)
    for (i in 1:length(acc)){
      k = which(m$gene == acc[i])
      
      tmp = m[k,k]
      scores[1+f,] = mean(tmp)
      
    }
  }
  
}

write.csv(m, file = unlist(snakemake@output[["scores"]]), row.names = F)