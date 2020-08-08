### HEADER ###
# HOTSPOT REGIONS
# description: check if similarity of a hotspot region to the respective protein differs from
#               a non-hotspot region
# input: hotspot/non-hotspot regions (extended version), human proteome embedding
# output: ...
# author: HR

library(dplyr)
library(stringr)
library(seqinr)

library(ggplot2)
library(tidyr)

library(future)
library(foreach)
library(doParallel)

registerDoParallel(cores=availableCores())
# requires functions from 01_trainingData.R and 02_computeScores.R

### INPUT ###
prot.emb = read.csv("../RUNS/HumanProteome/word2vec_model/hp_sequence_repres_w5_d100_seq2vec-TFIDF.csv",
                    stringsAsFactors = F)
regions = read.csv("data/regions_ext_substr.csv", stringsAsFactors = F)

### MAIN PART ###
# only keep proteins with hotspots
prot.emb = prot.emb[prot.emb$Accession %in% regions$Accession, ]

# randomly select a subset of proteins
k = 1e02
prot.emb = prot.emb[sample(nrow(prot.emb), k), ]

results = data.frame(Accession = rep(NA, nrow(prot.emb)),
                     no_hotspots = rep(NA, nrow(prot.emb)),
                     no_n.hotspots = rep(NA, nrow(prot.emb)),
                     len_hotspots = rep(NA, nrow(prot.emb)),
                     len_n.hotspots = rep(NA, nrow(prot.emb)),
                     sim_hotspots = rep(NA, nrow(prot.emb)),
                     sim_n.hotspots = rep(NA, nrow(prot.emb)))


# for each protein, calculate the mean similarity between all hsp and non-hsp regions
pb = txtProgressBar(min = 0, max = nrow(prot.emb), style = 3)
dimRange = c(4:ncol(prot.emb))

for (i in 1:nrow(prot.emb)) {
  
  setTxtProgressBar(pb, i)
  results$Accession[i] = prot.emb$Accession[i]
  
  # get regions
  tmp.regions = regions[regions$Accession == prot.emb$Accession[i], ]
  tmp.regions = get_seq_repres(tmp.regions, out = paste0(prot.emb$Accession[i], ".csv"))
  
  tmp.hsp = tmp.regions[tmp.regions$label == "hotspot", ]
  tmp.n.hsp = tmp.regions[tmp.regions$label == "non_hotspot", ]
  
  results$no_hotspots[i] = nrow(tmp.hsp)
  results$no_n.hotspots[i] = nrow(tmp.n.hsp)
  results$len_hotspots[i] = nchar(tmp.hsp$region) %>% mean
  results$len_n.hotspots[i] = nchar(tmp.n.hsp$region) %>% mean
  
  # calculate similarity
  sim.hsp = rep(NA, nrow(tmp.hsp))
  for (h in 1:nrow(tmp.hsp)) {
    sim.hsp[h] = dot_product(v1 = prot.emb[i, dimRange],
                             v2 = tmp.hsp[h, c(7:ncol(tmp.hsp))])
  }
  
  sim.n.hsp = rep(NA, nrow(tmp.n.hsp))
  for (n in 1:nrow(tmp.n.hsp)) {
    sim.n.hsp[n] = dot_product(v1 = prot.emb[i, dimRange],
                             v2 = tmp.n.hsp[n, c(7:ncol(tmp.n.hsp))])
  }
  
  results$sim_hotspots[i] = sim.hsp %>% mean()
  results$sim_n.hotspots[i] = sim.n.hsp %>% mean()
  # results$KS_statistic[i] = comp_similarity(d.H = sim.hsp, d.N = sim.n.hsp)
  
}

# analysis
plot(density(results$sim_hotspots))
plot(density(results$sim_n.hotspots %>% na.omit()))

summary(results$sim_hotspots)
summary(results$sim_n.hotspots %>% na.omit())

ks.test(results$sim_hotspots,
        results$sim_n.hotspots %>% na.omit(),
        alternative = "less")

summary(results$len_hotspots)
summary(results$len_n.hotspots %>% na.omit())

summary(results$no_hotspots)
summary(results$no_n.hotspots %>% na.omit())

# plot
res = cbind(results$sim_hotspots, results$sim_n.hotspots) %>% na.omit() %>%
  as.data.frame()
names(res) = c("hsp", "non-hsp")
res = tidyr::gather(res)

ggplot(res, aes(x = value, color = key)) +
  geom_density() +
  ggtitle("cosine similarity of regions to protein") +
  theme_bw()


### OUTPUT ###
ggsave("data/similarity_REGION_PROTEIN.png", plot = last_plot(), dpi = "retina")
write.csv(results, "data/similarity_REGION_PROTEIN.csv", row.names = F)


