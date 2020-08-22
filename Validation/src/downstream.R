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

## tmp!
# input = list.files(path = "similarity/scores", pattern = "RData", recursive = T, full.names = T)

output = "scores_hp_200814_dot.csv"


### MAIN PART ###

# collect all scores
for(i in 1:length(input)) {

  print(input[i])
  
  nm = str_split(input[i], coll("/"), simplify = T)[, 3]
  nm = str_split(nm, coll("."), simplify = T)[, 1] %>%
    as.character()
  
  # load scores
  load(input[i])
  
  scores.df = plyr::ldply(scores)
  scores.df$state = rep(c("pre", "post"), nrow(scores.df)*0.5)
  scores.df$embedding = rep(nm, nrow(scores.df))
  
  names(scores.df)[1] = "ground_truth"
  
  ### OUTPUT ###
  # if output file alredy exists (not the 1st iteration) open it and append to it
  # else create empty data frame
  
  if(file.exists(output)){
    
    df = read.csv(output, stringsAsFactors = F)
    df = rbind(df, scores.df)
    write.csv(df, file = output, row.names = F)
    
  } else {
    
    df = scores.df
    write.csv(df, file = output, row.names = F)
    
  }
  
}


