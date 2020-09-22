### SPLICING PREDICTOR ###
# description: build and characterise adjacency matrices
# input: training data sets
# output: adjacency matrices for every substrate
# author: HR


library(dplyr)
library(stringr)
library(igraph)

# setwd("Documents/QuantSysBios/ProtTransEmbedding/Predictor/")

### INPUT ###
indices = read.csv("data/ids_hp-subs_v5k.csv", stringsAsFactors = F, header = F)
colnames(indices) = c("subword", "word_ID")
indices$subword = toupper(indices$subword)

loadData = function(fname = ""){
  data = read.csv(paste0("data/TRAIN_", fname, ".csv"), stringsAsFactors = F)
  
  # how many labels are 0 and 1?
  print("------ label distribution in different product types ------")
  print(table(data[, c("type", "label")]) / nrow(data))
  
  # how many tokens are covered by one substrate?
  subs = data$substrateID %>% unique() %>% as.character()
  no_tokens = rep(NA, length(subs))
  
  for (i in 1:length(subs)) {
    cnt = data[data$substrateID == subs[i], ]
    cnt_tokens = cnt$tokens %>% paste(sep = "", collapse = " ") %>%
      str_split_fixed(coll(" "), Inf) %>%
      paste() %>% as.numeric() %>% unique()
    
    no_tokens[i] = length(cnt_tokens)
  }
  
  print("------ number of tokens in vocabulary covered by substrates ------")
  print(summary(no_tokens))
  print("------ percentage: ------")
  print(summary((no_tokens / nrow(indices))*100))
  
  return(data)
}


data = loadData(fname = "allTP")

# check token overlap in train and test data
test.data = read.csv("data/TEST_allTP.csv", stringsAsFactors = F)
train.tokens = as.character(data$tokens) %>% paste(collapse = " ", sep = " ") %>%
  str_split_fixed(pattern = coll(" "), Inf) %>% as.numeric() %>% unique()
print(length(train.tokens))

test.tokens = as.character(test.data$tokens) %>% paste(collapse = " ", sep = " ") %>%
  str_split_fixed(pattern = coll(" "), Inf) %>% as.numeric() %>% unique()
print(length(test.tokens))

intersect(train.tokens, test.tokens) %>% length()

# data_4hrs = loadData(fname = "4hrs")
# data_20hrs = loadData(fname = "20hrs")


### MAIN PART ###
buildAdjacencyMatrix = function(data = "",  substrateID = "", observed_only = F, undirected = F){
  
  AM = matrix(nrow = nrow(indices), ncol = nrow(indices))
  colnames(AM) = indices$word_ID
  rownames(AM) = indices$word_ID
  AM[is.na(AM)] = 0
  
  # every node is connected to itself --> diagonal is 1
  diag(AM) = 1
  
  data = data[data$substrateID == substrateID, ]
  if (observed_only) {data = data[data$label == 1, ]}
  
  # iterate products and increment counts at respective connections
  for(i in 1:nrow(data)) {
    cnt_tokens = data$tokens[i] %>% str_split_fixed(coll(" "), Inf) %>% as.numeric()
    
    for (j in 1:(length(cnt_tokens)-1)) {
      AM[cnt_tokens[j], cnt_tokens[j+1]] = AM[cnt_tokens[j], cnt_tokens[j+1]] + 1
      
      if (undirected){
        AM[cnt_tokens[j+1], cnt_tokens[j]] = AM[cnt_tokens[j+1], cnt_tokens[j]] + 1
      }
    }
    
  }
  
  # normalise by sum
  AM = AM / sum(AM)
  
  # remove tokens that are not in current substrate
  cnt_tokens = data$tokens %>% paste(collapse = " ") %>%
    str_split_fixed(pattern = coll(" "), Inf) %>%
    as.numeric() %>% unique()
  
  rm = which(! seq(1, nrow(indices)) %in% cnt_tokens)
  AM = AM[-rm, ]
  AM = AM[, -rm]
  
  # graph = graph_from_adjacency_matrix(AM, mode = "directed")
  # plot.igraph(graph)
  
  return(AM)
}


allAdjacencyMatrices = function(data = "", fname = ""){
  
  substrateIDs = data$substrateID %>% unique() %>% as.character()
  
  for(s in 1:length(substrateIDs)) {
    print(substrateIDs[s])
    
    # adjacency matrix for all possible products
    AM_all = buildAdjacencyMatrix(data, substrateID = substrateIDs[s])
    AM_obs = buildAdjacencyMatrix(data, substrateID = substrateIDs[s], observed_only = T)
    
    write.csv(AM_all,
              paste0("data/adjacency_matrices/", fname, "_allPossible_", substrateIDs[s], ".csv"),
              row.names = T)
    write.csv(AM_obs,
              paste0("data/adjacency_matrices/", fname, "_obsOnly_", substrateIDs[s], ".csv"),
              row.names = T)
  }
}


### OUTPUT ###
if(!dir.exists("data/adjacency_matrices/")) {dir.create("data/adjacency_matrices/")}

allAdjacencyMatrices(data, fname = "allTP")
# allAdjacencyMatrices(data, fname = "4hrs")
# allAdjacencyMatrices(data, fname = "20hrs")


