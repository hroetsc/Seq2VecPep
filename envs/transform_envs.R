setwd("/home/hanna/Documents/QuantSysBios/ProtTransEmbedding/Snakemake")
library(yaml)
library(plyr)
library(dplyr)
library(stringr)

env_base = read_yaml("envs/environment_base.yml")
env_snakemake = read_yaml("envs/environment_lab.yml")

dep_base = env_base$dependencies %>% as.data.frame()
dep_base = t(dep_base[1,])
dep_snakemake = env_snakemake$dependencies %>% as.data.frame()

not = c()

for (i in 1:nrow(dep_base)){
  if(! dep_base[i,1] %in% dep_snakemake$.){
    not = c(not, dep_base[i,1])
  }
}
not

red_base = str_split_fixed(dep_base[,1], coll("="), Inf)[,1]
red_snakemake = str_split_fixed(dep_snakemake[,1], coll("="), Inf)[,1]

not_name = c()

for (i in 1:length(red_base)){
  if(! red_base[i] %in% red_snakemake){
    not_name = c(not_name, red_base[i])
  }
}

rem = not[which(str_split_fixed(not, coll("="), Inf)[,1] %in% not_name)] %>% as.data.frame()
