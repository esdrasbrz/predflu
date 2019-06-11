library(tidyverse)
library(igraph)
library(gganimate)
library(tidygraph)
library(ggraph)

# Prepara dados para enviar ao gephi 
# Analises serão feitas no gephi

plague <- read_csv("../twitter-data/antivax.csv")

plague$user_id <- as.factor(plague$user_id)

plague_info <- lookup_users(plague$user_id)

vax <- inner_join(plague_raw, plague_info)

vax %>%
  mutate(source = user)%>%
  mutate(target = user_id)%>%
  select(source, target) %>% 
  write_csv("../twitter-data/graph_antivax.csv")


# Pós Gephi

top_antivax <- read_csv("../twitter-data/antivax_gephi.csv") %>% top_n(n = 10, wt = eigencentrality)

top_antivax$user_id <- as.factor(top_antivax$Id)

vax %>% select(user_id, screen_name) %>% distinct() %>% inner_join(top_antivax) %>% view()

