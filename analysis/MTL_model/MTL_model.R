#2026.4.27

# Goal: we want to predict available soil moisture content (nFK, %; See below L74) at three different soil depths (0-20cm, 20-40cm and 40-60cm)
# Model configuration = Long-short term memory (LSTM) + multi task learning (MTL) + attention
# What's the reasoning behind this configuration:
# 1. LSTM models are effective for time-series analysis 
# 2. MTL processes tasks separately, such as predicting at three different soil depths
# 3. Attention adds contextual information to predictions, which LSTM  does not have


#_______________ Libraries _______________
library(keras3) 
library(dplyr) # data pipelines
library(ggplot2) # visualization
library(data.table) # load csv file
library(purrr) # loop through list
library(tidyr) # pivoting
library(tidymodels) # data preprocess
library(tfdatasets) # needed for handling tensors
library(tensorflow)


#_______________ Data load and division _______________
setwd("C:/Users/....")

data1<-fread("Rubo_data.csv") 

# There are three subsets: train (60%)/validation (20%)/test (20%)
# Typically it's recommended that subsets are ordered chronologically.
# e.g. Train: 2021 data -> Val: 2022 data -> Test: 2023 data
split_main <- group_initial_split(data1, group = id_all, prop = 0.6)
train_df   <- training(split_main)  # training set
temp_df    <- testing(split_main)
split_val_test <- group_initial_split(temp_df, group = id_all, prop = 0.5)
val_df  <- training(split_val_test) # Validation set
test_df <- testing(split_val_test) # Test set

# You can just use train/ val/ test.csv files

#_______________inputs and outputs_______________
# The followings are the candidate sets of input variables
# For more details, please check Rubo and Zinkernagel, (2025) "Enhancing the prediction of irrigation demand for open field vegetable crops in Germany through neural networks, transfer learning, and ensemble models"

# 25 features
F25 <- c("tage_seit_aussaat",  "Tmean_gradC",  "Tmean_th9_25_gradC", "Tmin_gradC", "Tmax_gradC", 
         "relLuftfeuchte_mean_prozent", "relLuftfeuchte_min_prozent", "relLuftfeuchte_max_prozent", 
         "globalstrahlung_Wh_m2", "windgeschwindigkeit_m_s", "tageslicht_h", "tageslicht_h_th9", "Tmean_gradC_sum", 
         "Tmean_th9_25_gradC_sum", "globalstrahlung_Wh_m2_sum", "niederschlag_mm_sum", "tageslicht_h_sum", 
         "tageslicht_h_th9_sum", "Ton_prozent", "Schluff_prozent", "Sand_prozent", "C_org_prozent", "ibi",  "irmi",
         "wasser_input")

# 19 features
F19<- c("globalstrahlung_Wh_m2_sum","globalstrahlung_Wh_m2", "Tmean_th9_25_gradC_sum",
        "rain_sum14", "rain_sum7","Sand_prozent","tage_seit_aussaat","tageslicht_h_th9_sum",
        "tageslicht_h_sum", "Ton_prozent","relLuftfeuchte_max_prozent",
        "water_sum14","water_sum7","relLuftfeuchte_mean_prozent","relLuftfeuchte_min_prozent",
        "Tmax_gradC","irmi","wasser_input","ibi")

# 7 features
F7<-c("globalstrahlung_Wh_m2_sum", "tageslicht_h_sum","irmi","ibi","water_sum14", 
      "water_sum7","wasser_input")


# We aimed to predict the next day of soil moisture from the previous "7" days 
# e.g. t-6, t-5, ..  t-1, t (inputs) => t+1 (prediction)
n_past  <- 7  # past 7 days
n_steps <- 35 # growing seasons which can vary by crops

lag_vars <- c("B0020_lag", "B2040_lag", "B4060_lag") # for the SM prediction, we need previous soil moisture data (most influential variables!)

# Three outputs (soil moisture at three depths: 0-20cm, 20-40cm and 40-60cm) -> target variables
names_y <- c(
  "B0020_nFK_prozent", 
  "B2040_nFK_prozent",
  "B4060_nFK_prozent")
# nFK => available soil water content = (current water content - wilting point)/ (field capacity - wilting point) x 100
# https://www.dwd.de/DE/fachnutzer/landwirtschaft/dokumentationen/allgemein/basis_bodenfeuchte_doku.html;jsessionid=F0A5887A3EF9A8E161B1DB70095F0C6D.live21072?nn=344238


#_______________ Data preprocess _______________
# We used tidymodel framework (e.g. recipe) for this process
# For more details, read: https://www.tidymodels.org/start/recipes/

# 1. Normalization
# 2. Add previous SM data into our inputs
# 3. Remove redundant variables
rec <- recipe(train_df) %>% 
  step_mutate(
    B0020_lag = B0020_nFK_prozent, # Previous soil moisture data, which will be input variables
    B2040_lag = B2040_nFK_prozent,
    B4060_lag = B4060_nFK_prozent,
  ) %>%
  update_role(all_of(names_y), new_role = "outcome") %>% # three outputs
  update_role(-all_of(names_y), new_role = "predictor") %>% # inputs
  step_zv(all_predictors()) %>%  # remove variables that contain only a single value
  update_role(id_all, new_role = "id") %>% # In our data set, we have column "id", which doesn't need to be included in the analysis
  step_normalize(all_predictors()) %>% #normalization
  step_normalize(all_outcomes()) %>%
  prep()

#Execution of data preprocess
train_baked <- bake(rec, new_data = train_df) %>% select(all_of(c(F25,"id_all",names_y,lag_vars)))  # doesn't have to be F25...
val_baked   <- bake(rec, new_data = val_df)%>% select(all_of(c(F25,"id_all",names_y,lag_vars)))
test_baked  <- bake(rec, new_data = test_df)%>% select(all_of(c(F25,"id_all",names_y,lag_vars)))


# Given that this is a time-series analysis, we need to change the pre-processed data structure according to it
# The basic data structure for the time-series analysis is as follows....
# e.g. x: 1-7 -> y: 8; 2-8 -> y: 9... 28-34 -> y: 35
make_dataset <- function(baked_df, shuffle = FALSE) {
  pred_cols   <- which(!names(baked_df) %in% c(names_y, "id_all"))
  target_cols <- match(names_y, names(baked_df))
  
  # Loop for each id_all
  map(unique(baked_df$id_all), ~{ 
    df_id <- baked_df[baked_df$id_all == .x, , drop = FALSE]
    
    x <- head(as.matrix(df_id[, pred_cols]), -1)
    y <- tail(as.matrix(df_id[, target_cols]), -7)
    
    ds_x  <- timeseries_dataset_from_array(
      data = x, targets = NULL,
      sequence_length = 7 # 7 days we looked back 
      , sampling_rate = 1, 
      sequence_stride = 1, 
      shuffle = shuffle,
      batch_size = 16L # doesn't need to be 16L
    )
    # These tensor slices ensures that MTL can handle our three different tasks (three soil depths) separately
    ds_y1 <- tensor_slices_dataset(y[, 1, drop = FALSE]) %>% dataset_batch(16L)
    ds_y2 <- tensor_slices_dataset(y[, 2, drop = FALSE]) %>% dataset_batch(16L)
    ds_y3 <- tensor_slices_dataset(y[, 3, drop = FALSE]) %>% dataset_batch(16L)
    
    zip_datasets(list(ds_x, zip_datasets(list(ds_y1, ds_y2, ds_y3))))
  }) %>% reduce(dataset_concatenate)
}

final_train <- make_dataset(train_baked, shuffle = FALSE) # doesn't need to be shuffle= F 
final_val   <- make_dataset(val_baked,  shuffle = FALSE)
final_test  <- make_dataset(test_baked, shuffle = FALSE)

#__________________Model configuration__________________
# Customize our model class! (LSTM+MTL+attention)
# Parameter definition -> Feed forward -> Loss function -> compile -> training
# Hyperparameters such as dropout rate, units can be adjusted using BayesOpt or NSGA-II... etc

MTL_LSTM_UWFL_attention <- new_model_class(
  classname = "MTL_LSTM_UWFL",
  
  #__________________Define parameters__________________
  initialize = function(n_tasks = 3, time_steps = 7, ...) {
    super$initialize(...)
    self$n_tasks    <- as.integer(n_tasks)
    self$time_steps <- as.integer(time_steps)
    
    # Shared encoder (LSTM layers)
    self$lstm1   <- layer_lstm(units = 223, return_sequences = TRUE)
    self$lstm2   <- layer_lstm(units = 101, return_sequences = TRUE)  
    self$dropout <- layer_dropout(rate = 0.14) #0.3
    
    # Attention layers 
    self$attn_dense1 <- layer_dense(units = 101, activation = "tanh") 
    self$attn_dense2 <- layer_dense(units = 1, use_bias = FALSE)       
    
    # MTL: Task-specific heads  (n=3)
    self$head1 <- keras_model_sequential(layers = list(
      layer_dense(units = 32, activation = "relu"),
      layer_dense(units = 16, activation = "relu"),
      layer_dense(units = 1,  name = "B0020")
    ))
    self$head2 <- keras_model_sequential(layers = list(
      layer_dense(units = 32, activation = "relu"),
      layer_dense(units = 16, activation = "relu"),
      layer_dense(units = 1,  name = "B2040")
    ))
    self$head3 <- keras_model_sequential(layers = list(
      layer_dense(units = 32, activation = "relu"),
      layer_dense(units = 16, activation = "relu"),
      layer_dense(units = 1,  name = "B4060")
    ))
    
    self$log_vars <- self$add_weight( #UWFL Learnable log variance per task (See below L226)
      name        = "log_vars",
      shape       = shape(n_tasks),
      initializer = "zeros",
      trainable   = TRUE
    )
  },
  
  #__________________Feed forward__________________
  call = function(inputs, training = FALSE, ...) { 
    
    #Shared encoder
    x <- inputs %>%
      self$lstm1(training = training) %>%   
      self$lstm2(training = training)        
    
    # Attention mechanism => Bahdanau et al., 2014: "Neural Machine Translation by Jointly Learning to Align and Translate"
    # Step 1: eₖ = u^T * tanh(Wh * hk + bh)
    e <- x %>%
      self$attn_dense1() %>%    
      self$attn_dense2()         
    
    # Step 2: squeeze + softmax → αᵢ (attention weights)
    e_squeezed <- op_squeeze(e, axis = -1L)       
    alpha       <- op_softmax(e_squeezed, axis = -1L)  
    alpha_expanded <- op_expand_dims(alpha, axis = -1L) 
    
    # Step 3: Context vector:  weighted sum (attention weight * input)
    context <- op_sum(x * alpha_expanded, axis = 2L)     # Keras 3 use 1-based system so 2L not 1L
    #collapsing_sum (information synthesis) across timesteps -> look at all time steps simultaneously (important step higher value)
    
    context <- self$dropout(context, training = training)
    
    #Task heads
    out1 <- self$head1(context, training = training)
    out2 <- self$head2(context, training = training)
    out3 <- self$head3(context, training = training)
    
    list(out1, out2, out3)
  }, 
  
  #__________________uncertainty-weighted loss function (UWLF)__________________
  # Wang et al., 2025: Multi-task learning model driven by climate and remote sensing data collaboration for mid-season cotton yield prediction
  # The overall loss is easily dominated by one task, and ultimately, the loss of other tasks cannot affect the learning process of the
  # network-sharing layer. To address this, an uncertainty-weighted loss function (UWLF) was introduced
  
  compute_loss = function(x = NULL, y = NULL, y_pred = NULL, ...) {
    
   
    total_loss <- 0.0
    for (t in 1:self$n_tasks) {
      y_true_t  <- op_cast(y[[t]], dtype = "float32")
      mse_t     <- op_mean(op_square(y_true_t - y_pred[[t]]))
      precision <- op_exp(-self$log_vars[t])
      task_loss <- 0.5 * precision * mse_t + 0.5 * self$log_vars[t]
      total_loss <- total_loss + task_loss
    }
    total_loss
  }
)

#__________________Model compile__________________
model <- MTL_LSTM_UWFL_attention(n_tasks = 3)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001), #0.0001478527 (BayesOpt)
  metrics   = list(
    list(metric_mean_absolute_error(name = "B0020_mae")),
    list(metric_mean_absolute_error(name = "B2040_mae")),
    list(metric_mean_absolute_error(name = "B4060_mae"))
  )
)


callbacks <- list(
  callback_model_checkpoint(
    filepath       = "MTL_Geumuse.keras",
    monitor        = "val_loss",
    save_best_only = TRUE
  ),
  callback_early_stopping(
    monitor             = "val_loss",
    patience            = 10,
    min_delta           = 0.001,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor  = "val_loss",
    factor   = 0.5, #It automatically halves the learning rate whenever `val_loss` stops improving for 5 epochs
    patience = 5
  )
)

#__________________Model training__________________
history <- model %>% fit(
  final_train,
  epochs          = 300,
  validation_data = final_val,
  shuffle         = FALSE,
  callbacks       = callbacks
)

res <- evaluate(model, final_test) # Use "test" set!

sprintf("B0020 MAE: %.3f", res$B0020_mae)
sprintf("B2040 MAE: %.3f", res$B2040_mae)
sprintf("B4060 MAE: %.3f", res$B4060_mae)

file.rename("MTL_Geumuse.keras", "MLT_attention_v3.keras")


#__________________Model evaluation__________________
# We used iteration method: Predictions are iteratively fed back into the model
# Iterative method can easily generalize the one-step-ahead prediction model to multi-step prediction (Lim and Zohren, 2021)
# This method is very susceptible to error accumulation.

all_ids <- test_df %>% distinct(id_all) %>% pull(id_all)
#all_ids

all_metrics <- purrr::map_dfr(all_ids, function(id) {
  
  # filter one id
  id_test <- test_df %>% filter(id_all == id)
  
  # Set initial input data for our trained model
  x_t <- bake(rec, id_test) %>%
    select(all_of(c(F25, lag_vars))) %>%
    as.data.frame()
  
  nn    <- which(colnames(x_t) %in% lag_vars)
  preds <- matrix(NA, nrow = n_steps, ncol = length(names_y))
  
  # Iteration loop during the growing season
  for (i in seq_len(n_steps)) {
    X_tensor <- array(
      as.matrix(x_t[(1:n_past) + i - 1, ]),
      dim = c(1, n_past, ncol(x_t))
    )
    pred       <- as.numeric(model %>% predict(X_tensor, verbose = 0))
    preds[i, ] <- pred
    x_t[n_past + i, nn] <- pred # The predicted SM will be plugged back into the next input SM variables (autoregressive)
  }
  
  # Inverse transformation
  pred_df <- as.data.frame(preds)
  colnames(pred_df) <- names_y
  
  stats <- tidy(rec, number = 4) %>%
    pivot_wider(names_from = statistic, values_from = value) #mean and standard deviation from train set
  
  pred_real <- pred_df %>% #predictd values
    mutate(
      B0020_nFK_prozent = (B0020_nFK_prozent * stats$sd[1]) + stats$mean[1],
      B2040_nFK_prozent = (B2040_nFK_prozent * stats$sd[2]) + stats$mean[2],
      B4060_nFK_prozent = (B4060_nFK_prozent * stats$sd[3]) + stats$mean[3]
    )
  
  obs_df <- id_test[(n_past + 1):(n_past + n_steps), ] %>% #observed values
    select(all_of(names_y))
  
  results <- bind_cols(
    pred_real %>% rename_with(~ paste0(.x, "_pred")),
    obs_df    %>% rename_with(~ paste0(.x, "_obs"))
  )
  
  # Evaluation metrics (n=3): RMSE, nRMSE and d
  
cat(" nRMSE ≤ 10%: “excellent”
  10 % < nRMSE ≤ 20%: “good” 
  20% <  nRMSE ≤ 30%: “fair”
         nRMSE > 30%: “poor”")

 cat("  d ≥ 0.9: “excellent”
  0.8 ≤ d < 0.9: “good” 
  0.7 ≤ d< 0.8: “fair” 
        d < 0.7: “poor”")
  
  purrr::map_dfr(names_y, function(v) {
    truth    <- results[[paste0(v, "_obs")]]
    estimate <- results[[paste0(v, "_pred")]]
    valid    <- !is.na(truth) & !is.na(estimate)
    truth    <- truth[valid]
    estimate <- estimate[valid]
    
    rmse     <- rmse_vec(truth = truth, estimate = estimate)
    nrmse    <- 100 * rmse / mean(truth)
    mean_obs <- mean(truth)
    d_index  <- 1 - sum((estimate - truth)^2) /
      sum((abs(estimate - mean_obs) + abs(truth - mean_obs))^2)
    
    tibble(id_all = id, variable = v, RMSE = rmse, nRMSE = nrmse, d_index = d_index)
  })
})


avg_metrics <- all_metrics %>%
  group_by(variable) %>%
  summarise(
    RMSE    = mean(RMSE,    na.rm = TRUE),
    nRMSE   = mean(nRMSE,   na.rm = TRUE),
    d_index = mean(d_index, na.rm = TRUE),
    n_ids   = n()
  )

avg_metrics




all_ids <- test_df %>% distinct(id_all) %>% pull(id_all)

# 1. Extract normalization stats ONCE outside the loop to improve performance
stats <- tidy(rec, number = 4) %>%
  pivot_wider(names_from = statistic, values_from = value)

# 2. Run the prediction loop ONLY ONCE to get a master dataset
all_results_df <- purrr::map_dfr(all_ids, function(id) {
  
  id_test <- test_df %>% filter(id_all == id)
  
  # Ensure there are enough rows to predict
  if (nrow(id_test) < n_past + n_steps) return(NULL)
  
  x_t <- bake(rec, id_test) %>%
    select(all_of(c(F13, lag_vars))) %>%
    as.data.frame()
  
  nn    <- which(colnames(x_t) %in% lag_vars)
  preds <- matrix(NA, nrow = n_steps, ncol = length(names_y))
  
  # Autoregressive iteration
  for (i in seq_len(n_steps)) {
    X_tensor <- array(
      as.matrix(x_t[(1:n_past) + i - 1, ]),
      dim = c(1, n_past, ncol(x_t))
    )
    pred       <- as.numeric(model %>% predict(X_tensor, verbose = 0))
    preds[i, ] <- pred
    x_t[n_past + i, nn] <- pred
  }
  
  # Inverse transform and name columns
  pred_df <- as.data.frame(preds) %>% 
    setNames(names_y) %>%
    mutate(
      MP_20 = (MP_20 * stats$sd[1]) + stats$mean[1],
      MP_45 = (MP_45 * stats$sd[2]) + stats$mean[2],
      MP_70 = (MP_70 * stats$sd[3]) + stats$mean[3]
    )
  
  obs_df <- id_test[(n_past + 1):(n_past + n_steps), ] %>%
    select(all_of(names_y))
  
  # Bind predictions and observations, adding tracking columns
  bind_cols(
    pred_df %>% rename_with(~ paste0(.x, "_pred")),
    obs_df  %>% rename_with(~ paste0(.x, "_obs"))
  ) %>%
    mutate(step = row_number(), id_all = as.character(id))
})


# 3. Pivot the data long for plotting
plot_df_long <- all_results_df %>%
  pivot_longer(
    -c(step, id_all),
    names_to      = c("variable", "type"),
    names_pattern = "(.+)_(pred|obs)"
  )


# 4. Calculate metrics smoothly using the pivoted dataframe
all_metrics <- plot_df_long %>%
  pivot_wider(names_from = type, values_from = value) %>%
  filter(!is.na(obs) & !is.na(pred)) %>%
  group_by(id_all, variable) %>%
  summarise(
    RMSE     = rmse_vec(truth = obs, estimate = pred),
    mean_obs = mean(obs),
    nRMSE    = 100 * RMSE / mean_obs,
    d_index  = 1 - sum((pred - obs)^2) / sum((abs(pred - mean_obs) + abs(obs - mean_obs))^2),
    .groups  = "drop"
  ) %>%
  select(-mean_obs) # Drop intermediate column to match your exact previous output

avg_metrics <- all_metrics %>%
  group_by(variable) %>%
  summarise(
    RMSE    = mean(RMSE,    na.rm = TRUE),
    nRMSE   = mean(nRMSE,   na.rm = TRUE),
    d_index = mean(d_index, na.rm = TRUE),
    n_ids   = n()
  )

avg_metrics


# 5. Plot
ggplot(plot_df_long, aes(x = step, y = value, colour = type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2) +
  facet_grid(variable ~ id_all) +
  scale_colour_manual(
    values = c(obs = "#2C3E50", pred = "#E74C3C"),
    labels = c(obs = "Observed", pred = "Predicted")
  ) +
  scale_y_continuous(limits = c(-100, 950), breaks = seq(0, 950, 100)) +
  labs(
    title    = "Predicted vs Observed — All test IDs",
    x        = "Days after planting",
    y        = "Matric potential",
    colour   = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    strip.text.y    = element_text(angle = 0, size = 9),
    strip.text.x    = element_text(size = 8)
  )


