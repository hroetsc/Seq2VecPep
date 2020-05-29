library(plyr)
library(dplyr)
library(tidyr)
library(tidymodels)
library(tidytext)
library(forcats)
library(readr)
library(embed)


seqs = readr::read_csv("../../RUNS/HumanProteome/results/sequence_repres_w3_d100_seq2vec.csv")

# PCA

pca_rec = recipe(~., data = seqs) %>%
  update_role(Accession, seqs, tokens, new_role = "id") %>% # keep columns as id columns
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors())

pca_prep = prep(pca_rec)
pca_prep

tidied_pca = tidy(pca_prep, 2)

pdf("../human_proteome/results/PCs.pdf")
tidied_pca %>%
  filter(component %in% paste0("PC", 1:5)) %>%
  mutate(component = fct_inorder(component)) %>%
  ggplot(aes(value, terms, fill = terms)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~component, nrow = 1) +
  labs(y = NULL)
dev.off()

tidied_pca %>%
  filter(component %in% paste0("PC", 1:4)) %>%
  group_by(component) %>%
  top_n(8, abs(value)) %>%
  ungroup() %>%
  mutate(terms = reorder_within(terms, abs(value), component)) %>%
  ggplot(aes(abs(value), terms, fill = value > 0)) +
  geom_col() +
  facet_wrap(~component, scales = "free_y") +
  scale_y_reordered() +
  labs(
    x = "Absolute value of contribution",
    y = NULL, fill = "Positive?"
  )

juice(pca_prep) %>%
  ggplot(aes(PC1, PC2)) +
  geom_point(alpha = 0.7, size = 2)+
  labs(color = NULL)


# UMAP
umap_rec = recipe(~., data = seqs) %>%
  update_role(Accession, seqs, tokens, new_role = "id") %>%
  step_normalize(all_predictors()) %>%
  step_umap(all_predictors())

umap_prep = prep(umap_rec)

umap_prep

juice(umap_prep) %>%
  ggplot(aes(umap_1, umap_2)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(color = NULL)
