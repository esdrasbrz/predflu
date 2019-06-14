library(tidyverse)
library(rtweet)

token <- create_token(
  app = "rtweet_token",
  consumer_key = "",
  consumer_secret = "")


vax <- search_tweets(q = "vaccines AND chemicals lang:en", n = 10000)
vax2 <- search_tweets(q = "#antivax lang:en", n = 10000)

users <- vax$user_id 
write_csv(users, "../twitter-data/antivaxxers.csv")

pro_plague <- get_friends(users[1:15])
pro_plague2 <- get_friends(users[16:30])
pro_plague3 <- get_friends(users[31:45])
pro_plague4 <- get_friends(users[46:60])
pro_plague5 <- get_friends(users[61:75])
pro_plague6 <- get_friends(users[76:90])
pro_plague7 <- get_friends(users[91:105])
pro_plague8 <- get_friends(users[106:120])


plague <- rbind(pro_plague, pro_plague2, pro_plague3, pro_plague4, pro_plague5, pro_plague6, pro_plague7, pro_plague8)
write_csv(plague, "../twitter-data/antivax.csv")
