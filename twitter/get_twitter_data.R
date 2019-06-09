library(tidyverse)
library(rtweet)
  
token <- create_token(
  app = "rtweet_token",
  consumer_key = "",
  consumer_secret = "")

flu_2009 <- read_tsv("../twitter-data/RelatedVsNotRelated2009TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "related" = X2) %>% distinct()

flu_2012 <- read_tsv("../twitter-data/RelatedVsNotRelated2012TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "related" = X2) %>% distinct()

flu_2009$status_id <- as.character(flu_2009$status_id)
flu_2012$status_id <- as.character(flu_2012$status_id)

tweets_2009 <- lookup_tweets(statuses = flu_2009$status_id) %>% select(status_id, created_at, text)
tweets_2012 <- lookup_tweets(statuses = flu_2012$status_id) %>% select(status_id, created_at, text)

tweets_2009 <- left_join(tweets_2009, flu_2009)
tweets_2012 <- left_join(tweets_2012, flu_2012)

infection_2009 <- read_tsv("../twitter-data/AwarenessVsInfection2009TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "infection" = X2) %>% distinct()

infection_2012 <- read_tsv("../twitter-data/AwarenessVsInfection2012TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "infection" = X2) %>% distinct()

infection_2009$status_id <- as.character(infection_2009$status_id)
infection_2012$status_id <- as.character(infection_2012$status_id)

tweets_2009 <- left_join(tweets_2009, infection_2009)
tweets_2012 <- left_join(tweets_2012, infection_2012)

self_2009 <- read_tsv("../twitter-data/SelfVsOthers2009TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "self" = X2) %>% distinct()

self_2012 <- read_tsv("../twitter-data/SelfVsOthers2012TweetIDs.txt", col_names = FALSE) %>%
  rename("status_id" = X1, "self" = X2) %>% distinct()

self_2009$status_id <- as.character(self_2009$status_id)
self_2012$status_id <- as.character(self_2012$status_id)

tweets_2009 <- left_join(tweets_2009, self_2009)
tweets_2012 <- left_join(tweets_2012, self_2012)

tweets <- rbind(tweets_2009, tweets_2012) %>% select(created_at, text, related, infection, self)

tweets %>% write_csv("../twitter-data/tweets.csv")

