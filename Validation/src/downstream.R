### HEADER ###
# EVALUATION OF DIFFERENT EMBEDDING METHODS
# description:  collect all scores for the current iteration and
#               append it to overall output file
# author:       HR


library(stringr)
library(plyr)
library(dplyr)
library(tidyr)


print("### CONCATENATE SCORES ###")


### INPUT ###
input = snakemake@input[["scores"]]
output = "scores_hst.csv"

# if output file alredy exists (not the 1st iteration) open it and append to it
# else create empty data frame

if(file.exists(output)){

  df = read.csv(file = output, stringsAsFactors = F, header = T)
  idx = nrow(df) + 1

} else {

  df = matrix(ncol = 14, nrow = length(input)) %>% as.data.frame()
  idx = 1

  colnames(df) = c("iteration", "embedding",
                   "syntax_diff", "syntax_SD", "syntax_R2",
                   "MF_semantics_diff", "MF_semantics_SD", "MF_semantics_R2",
                   "BP_semantics_diff", "BP_semantics_SD", "BP_semantics_R2",
                   "CC_semantics_diff", "CC_semantics_SD", "CC_semantics_R2")

}


### MAIN PART ###

# collect all scores
for(i in 1:length(input)) {

  print(input[i])

  tmp = read.table(file = input[i], sep = " ", stringsAsFactors = F, header = T)

  # get embedding
  emb = str_split(input[i], coll("/"), simplify = T)[,3]
  df[idx, "embedding"] = str_split(emb, coll("."), simplify = T)[,1] %>% as.character()

  # get the number of iteration
  df[idx, "iteration"] = length(which(df$embedding[idx] %in% df$embedding))

  # extract scores - this is a bit dirty
  df$syntax_diff[idx] = tmp[1,1]
  df$syntax_SD[idx] = tmp[2,1]
  df$syntax_R2[idx] = tmp[3,1]

  df$MF_semantics_diff[idx] = tmp[1,2]
  df$MF_semantics_SD[idx] = tmp[2,2]
  df$MF_semantics_R2[idx] = tmp[3,2]

  df$BP_semantics_diff[idx] = tmp[1,3]
  df$BP_semantics_SD[idx] = tmp[2,3]
  df$BP_semantics_R2[idx] = tmp[3,3]

  df$CC_semantics_diff[idx] = tmp[1,4]
  df$CC_semantics_SD[idx] = tmp[2,4]
  df$CC_semantics_R2[idx] = tmp[3,4]

  idx = idx + 1
}


### OUPUT ###

write.csv(df, file = output, row.names = F)
