library(tidyverse)
library(keras)


# Read data
flu <- read_csv("../twitter-data/tweets.csv")
flu_2019 <- read_csv("../twitter-data/tweets_2019.csv")

# Set targets as NA
flu_2019$related <- NA
flu_2019$infection <- NA
flu_2019$self <- NA


flu_all <- rbind(flu, flu_2019)
text <- flu_all$text

# Set 1000 as max numbers of words as factors
max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)

# Tokenize words
tokenizer %>% 
  fit_text_tokenizer(text)

text_seqs <- texts_to_sequences(tokenizer, text)


# Parametros Keras
maxlen <- 100
batch_size <- 16
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 10


# Define train as 2009 flu data
x_train <- text_seqs[1:nrow(flu)] %>%
  pad_sequences(maxlen = maxlen)
y_train <- flu$related


# Define Keras model
model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


# Train model
hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1
  )


# Set test set as 2019 flu data
x_test <- text_seqs[-(1:nrow(flu))] %>%
  pad_sequences(maxlen = maxlen)

# Make predictions
pred <- predict(model, x_test, batch_size = batch_size)
flu_2019$related <- ifelse(pred > 0.65, 1, 0)

