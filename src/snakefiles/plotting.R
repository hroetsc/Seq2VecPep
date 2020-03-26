### HEADER ###
# PROTEIN/TRANSCRIPT EMBEDDING FOR ML/DEEP LEARNING
# description:  dimension reduction and plotting
# input:        protein matrices
# output:       some nice plots
# author:       HR

# tmp!!!
# protein.repres = read.csv(file = "results/embedded_proteome/proteome_repres.csv", stringsAsFactors = F, header = T)
# protein.repres.random = read.csv(file = "results/embedded_proteome/proteome_repres_random.csv", stringsAsFactors = F, header = T)
# PropMatrix = read.csv(file = "data/peptidome/biophys_properties.csv", stringsAsFactors = F, header = T)

print("### DIMENSION REDUCTION / PLOTTING ###")

library(seqinr)
library(protr)
library(Peptides)
library(plyr)
library(dplyr)
library(stringr)
library(readr)
library(uwot)
library(ggplot2)
library(grDevices)
library(RColorBrewer)

### INPUT ###
protein.repres = read.csv(file = snakemake@input[["proteome_repres"]], stringsAsFactors = F, header = T)
protein.repres.random = read.csv(file = snakemake@input[["proteome_repres_random"]], stringsAsFactors = F, header = T)
PropMatrix = read.csv(file = snakemake@input[["properties"]], stringsAsFactors = F, header = T)

# define range in which the embedding dimensions are
if ("TF_IDF_score" %in% colnames(protein.repres)) {
  dim_range = c(which(colnames(protein.repres)=="TF_IDF_score")+1, ncol(protein.repres))
} else {
  dim_range = c(which(colnames(protein.repres)=="tokens")+1, ncol(protein.repres))
}
colnames(protein.repres)[c(dim_range[1]:dim_range[2])] = seq(1, (dim_range[2]-dim_range[1]+1))
colnames(protein.repres.random)[c(dim_range[1]:dim_range[2])] = seq(1, (dim_range[2]-dim_range[1]+1))

### MAIN PART ###
print("APPLY UMAP TO EMBEDDINGS")
set.seed(42)

print('trained weights')
dims_UMAP = umap(protein.repres[,c(dim_range[1]:ncol(protein.repres))],
                 n_neighbors = 5,
                 min_dist = 0.01,
                 n_trees = 50,
                 verbose = T,
                 approx_pow = T,
                 ret_model = T,
                 metric = "euclidean",
                 scale = "none",
                 n_epochs = 500,
                 n_threads = 11)

umap_coords <- data.frame("X1"=dims_UMAP$embedding[,1],"X2"=dims_UMAP$embedding[,2])
proteinsUMAP <- cbind(umap_coords, protein.repres)

print('random weights')
dims_UMAP.random = umap(protein.repres.random[,c(dim_range[1]:ncol(protein.repres.random))],
                 n_neighbors = 5,
                 min_dist = 0.01,
                 n_trees = 50,
                 verbose = T,
                 approx_pow = T,
                 ret_model = T,
                 metric = "euclidean",
                 scale = "none",
                 n_epochs = 500,
                 n_threads = 11)

umap_coords.random <- data.frame("X1"=dims_UMAP.random$embedding[,1],"X2"=dims_UMAP.random$embedding[,2])
proteinsUMAP.random <- cbind(umap_coords.random, protein.repres.random)

# concatenate dataframes
proteinsUMAP.Props = left_join(proteinsUMAP, PropMatrix)
proteinsUMAP.Props = na.omit(proteinsUMAP.Props)
proteinsUMAP.Props.random = left_join(proteinsUMAP.random, PropMatrix)
proteinsUMAP.Props.random = na.omit(proteinsUMAP.Props.random)

### PLOTTING ###
print("GENERATE PLOTS")
# use base R because ggplot is not recalculating the color code
# colour gradient
spectral <- RColorBrewer::brewer.pal(10, "Spectral")
color.gradient = function(x, colsteps=1000) {
  return( colorRampPalette(spectral) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}
# size gradient
size.gradient = function(x, r_min = min(x), r_max = max(x), t_min = 0.5, t_max = 1.5) {
  return((((x-r_min)/(r_max - r_min)) * (t_max - t_min)) + t_min)
}

# plotting function
plotting = function(prop = "", data = "", random = ""){
  if (random == T) {
    title = paste0("random embeddings, by ", as.character(prop))
    label = "_random"
  } else {
    title = paste0("embedded proteome, by ", as.character(prop))
    label = ""
  }

  png(filename = paste0("results/plots/",
                        str_replace_all(as.character(prop), coll("."), coll("_")), label,".png"),
      width = 1500, height = 1500, res = 300)
  print(plot(data$X2 ~ data$X1,
              col = color.gradient(data[, prop]),
              cex = size.gradient(data[, "rPCP"]),
              xlab = "UMAP 1", ylab = "UMAP 2",
              main = title,
              sub = "red: low, blue: high, size: rPCP",
              cex.sub = 0.8,
              pch = 1))
  dev.off()
}

# remove output directory
# if (dir.exists(unlist(snakemake@output[["plot"]]))){
#   unlink(unlist(snakemake@output[["plot"]]),
#          recursive = T, force = T)
# }

for (i in (dim_range[2]+3):ncol(proteinsUMAP.Props)){
  plotting(prop = colnames(proteinsUMAP.Props)[i], data = proteinsUMAP.Props, random = F)
  plotting(prop = colnames(proteinsUMAP.Props.random)[i], data = proteinsUMAP.Props.random, random = T)
}

# check if all output files have been written to the directory
# retrieve file names in output directory
# files = list.files(path = unlist(snakemake@output[["plot"]]),
#                    pattern = ".png", all.files = T, full.names = T)
# file_list = rep(NA, length(files))
#
# for (f in 1:length(files)) {
#   file_list[f] = str_split(files[f], pattern = coll("/"), simplify = T)[3]
#   file_list[f] = str_split(file_list[f], pattern = coll("."), simplify = T)[1]
# }
#
# # compare files in directory with properties
# for (c in 2:ncol(PropMatrix)){
#   if(! str_replace_all(colnames(PropMatrix)[c], coll("."), coll("_")) %in% file_list) {
#     print(paste0("WARNING: no plot for property ",
#                  str_replace_all(colnames(PropMatrix)[c], coll("."), coll("_")),
#                  " found!"))
#   }
# }

### OUTPUT ###
# proteins with biophysical properties
write.csv(proteinsUMAP.Props, file = unlist(snakemake@output[["proteome_props"]]))
write.csv(proteinsUMAP.Props.random, file = unlist(snakemake@output[["proteome_props_random"]]))

# tmp!
# write.csv(proteinsUMAP.Props, file = "results/embedded_proteome/protein_repres_props.csv")
# write.csv(proteinsUMAP.Props.random, file = "results/embedded_proteome/protein_repres_props_random.csv")
