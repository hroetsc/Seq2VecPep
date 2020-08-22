# SCOP: structural classification of proteins

library(dplyr)
library(tibble)
library(stringr)


### INPUT ###
# Load SCOP classification
scop <- readr::read_delim("../../../files/scop_cla_latest.txt", skip = 5, delim = " ")
scop_class <- scop$`SF-UNIREG` %>%
  str_remove_all(pattern = "TP=") %>%
  str_remove_all(pattern = "CL=") %>%
  str_remove_all(pattern = "CF=") %>%
  str_remove_all(pattern = "SF=") %>%
  str_remove_all(pattern = "FA=") %>% 
  str_split_fixed(pattern = ",", Inf)
scop_class <- as_tibble(scop_class) %>% as.data.frame()
scop_class = cbind(scop$`FA-PDBREG`, scop_class)
colnames(scop_class) <- c("Accession","TP", "CL", "CF", "SF", "FA")

# TP=protein type, CL=protein class, CF=fold, SF=superfamily, FA=family

# SCOP node descriptions
scop_des <- readr::read_delim("../../../files/scop_des_latest.txt", delim = " ")
