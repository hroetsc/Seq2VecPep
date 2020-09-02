### HEADER ###
# HOTSPOT PREDICTION
# description: generate feature set using AA-Index
# input: tokens
# output: propensity scales of tokens
# author: HR

library(plyr)
library(dplyr)
library(stringr)

library(bio3d)
library(seqinr)

embeddingDim = 128

### INPUT ###
tokenEmb = read.csv("data/token_embeddings.csv", stringsAsFactors = F)

### MAIN PART ###
# load AA indices and randomly select as many scales as embeddingDim
data("aaindex")

set.seed(42)
cnt_indices = sample(length(aaindex), embeddingDim)
cnt_props = names(aaindex)[cnt_indices]

token.AA = matrix(ncol = embeddingDim+1, nrow = nrow(tokenEmb))
token.AA[, 1] = tokenEmb$subword

pb = txtProgressBar(min = 0, max = nrow(token.AA), style = 3)

for (i in 1:nrow(token.AA)) {
  
  setTxtProgressBar(pb, i)
  
  aa = strsplit(tokenEmb$subword[i], "") %>%
    unlist()
  
  for (j in 1:length(cnt_props)) {
    
    if (length(aa) > 1){
      token.AA[i, j+1] = aa2index(aa = aa,
                                  index = cnt_props[j], window = 1) %>%
        mean()
    } else {
      
      id = aaindex[[cnt_props[j]]][["I"]] %>% names() %>% a()
      val = aaindex[[cnt_props[j]]][["I"]][id == aa]
      
      if (length(val) > 0) {
        token.AA[i, j+1] = val
      } else {
        token.AA[i, j+1] = 0
      }
      
    }
    
    
    
  }
  
}

token.AA = as.data.frame(token.AA)
names(token.AA) = c("subword", cnt_props)


### OUTPUT ###
write.csv(token.AA, "data/token_AAindices.csv", row.names = F)







