### SPLICING PREDICTOR ###
# description: generate all possible products from tokens of substrate
# input: database, tokens (human proteome), token embeddings
# output: all possible products with label
# author: HR

library(plyr)
library(dplyr)
library(stringr)

library(doParallel)
library(future)
registerDoParallel(availableCores())

# setwd("Documents/QuantSysBios/ProtTransEmbedding/Predictor/")

### INPUT ###
DB = read.csv("data/ProteasomeDB.csv",
              stringsAsFactors = F)
DB$spliceType[DB$spliceType == ""] = "PCP"
DB_4hrs = DB[DB$digestTime == 4, ]
DB_20hrs = DB[DB$digestTime != 4, ]

indices = read.csv("data/ids_hp-subs_v5k.csv", stringsAsFactors = F, header = F)
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)


### MAIN PART ###
########## generate all possible products ##########

# for every substrate --> get all possible PCPs and PSPs
# use tokens in human proteome
# prefer longer tokens over shorter ones

match2tokens = function(SR = "") {
  # all tokens that occur in current product / SR
  poss.tok = indices[str_detect(string = SR, pattern = indices$subword), ] %>%
    arrange(nchar(subword) %>% desc())
  
  # remove them from the SR starting with longest
  cnt.prod = SR
  IDs = rep(NA, nrow(poss.tok))
  
  for (e in 1:nrow(poss.tok)) {
    if (length(SR) > 0 & str_detect(SR, poss.tok$subword[e])) {
      SR = str_replace(string = SR, pattern = poss.tok$subword[e], replacement = "")
      IDs[e] = poss.tok$word_ID[e]
    }
  }
  
  # order them by occurence in SR
  IDs = na.omit(IDs)
  IDs = IDs[str_locate(cnt.prod, indices$subword[IDs])[, 1] %>% order()]
  IDs = paste(IDs, collapse = " ")
  
  return(IDs)
}



genPCP = function(subSeq = "") {
  print("all possible PCPs")
  
  subEnc = strsplit(subSeq, "") %>%
    unlist()
  
  t = length(subEnc)^2  # theoretical amount of PCPs
  pcp.products = rep(NA, t)
  tokenIDs = rep(NA, t)
  pos = rep(NA, t)
  
  pb = txtProgressBar(min = 0, max = length(subEnc), style = 3)
  counter = 1
  
  system.time(for (o in 1:length(subEnc)) {
    setTxtProgressBar(pb, o)
    
    for (p in 1:length(subEnc)) {
      
      if (! p < o) {
        cnt.pcp = paste(subEnc[o:p], collapse = "")
        pcp.products[counter] = cnt.pcp
        tokenIDs[counter] = match2tokens(SR = cnt.pcp)
        pos[counter] = paste(o, p, sep = "_")
        
        counter = counter + 1
      
      }
    }
  })[3]
  
  res = data.frame(product = pcp.products,
                   tokens = tokenIDs,
                   positions = pos,
                   type = rep("PCP", t)) %>% na.omit()
  
  res$product = as.character(res$product)
  
  return(res)
  
}


genPSP = function(subSeq = "", pcp.products = "") {
  print("all possible PSPs")
  
  reactants = pcp.products$product %>% as.character()
  
  t = length(reactants)^2
  
  cis.products = rep(NA, t)
  revCis.products = rep(NA, t)
  trans.products = rep(NA, t)
  tokenIDs = rep(NA, t)
  spl.pos = rep(NA, t)
  
  pb = txtProgressBar(min = 0, max = length(reactants), style = 3)
  counter = 1
  
  system.time(for (a in 1:length(reactants)) {
    setTxtProgressBar(pb, a)
    
    for (b in 1:length(reactants)) {
      
      cnt.psp = paste(reactants[a], reactants[b], sep = "")
      
      # get reactant position in substrate
      i1.j1 = str_locate(string = subSeq, pattern = reactants[a])
      i2.j2 = str_locate(string = subSeq, pattern = reactants[b])
      pos = c(i1.j1, i2.j2)
      
      spl.pos[counter] = paste(pos, collapse = "_")
      
      # use Gerd's code
      overlap <- any(pos[3]:pos[4] %in% pos[1]:pos[2])
      
      if (overlap) {
        trans.products[counter] = cnt.psp
        tokenIDs[counter] = paste(match2tokens(SR = reactants[a]),
                                  match2tokens(SR = reactants[b]),
                                  sep = " ")
        
      } else if (pos[2] > pos[3]) {
        revCis.products[counter] = cnt.psp
        tokenIDs[counter] = paste(match2tokens(SR = reactants[a]),
                                  match2tokens(SR = reactants[b]),
                                  sep = " ")
        
      } else if (pos[2] < pos[3]) {
        cis.products[counter] = cnt.psp
        tokenIDs[counter] = paste(match2tokens(SR = reactants[a]),
                                  match2tokens(SR = reactants[b]),
                                  sep = " ")
        
      }
      
      counter = counter + 1
      
    }
    
  })[3]
  
  cis = data.frame(product = cis.products,
                   tokens = tokenIDs,
                   positions = spl.pos,
                   type = rep("cis", t)) %>% na.omit()
  
  revCis = data.frame(product = revCis.products,
                      tokens = tokenIDs,
                      positions = spl.pos,
                      type = rep("revCis", t)) %>% na.omit()
  
  trans = data.frame(product = trans.products,
                      tokens = tokenIDs,
                      positions = spl.pos,
                      type = rep("trans", t)) %>% na.omit()
  
  return(rbind(cis, revCis, trans))
  
}


########## get labels ##########
getLabels = function(allPossible){
  print("assign labels to all possible peptides")
  
  allPossible = allPossible %>% mutate(label_all = ifelse(product %in% DB$pepSeq, 1, 0))
  allPossible = allPossible %>% mutate(label_4hrs = ifelse(product %in% DB_4hrs$pepSeq, 1, 0))
  allPossible = allPossible %>% mutate(label_20hrs = ifelse(product %in% DB_20hrs$pepSeq, 1, 0))
  
  return(allPossible)
}



########## apply ##########
subs = DB$substrateSeq %>% unique()
master = list()
master = foreach(s = 1:length(subs)) %dopar% {
  
  pcp.products = genPCP(subSeq = subs[s])
  psp.products = genPSP(subSeq = subs[s], pcp.products)
  
  # clean PCP products
  pcp.products = pcp.products[nchar(pcp.products$product) > 1, ]
  pcp.products = pcp.products[-which(pcp.products$product == subs[s]), ]
  
  allPossible = rbind(pcp.products, psp.products)
  
  master[[s]] = getLabels(allPossible)
  
}

names(master) = subs


### OUTPUT ###
save(master, file = "data/allPossible.RData")



