
library(keras3)
library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(tidyr)
library(tidymodels)
#library(zoo)
library(tfdatasets)
library(tensorflow)

F_fine_no_crop<- c("globalstrahlung_Wh_m2", "Tmean_gradC", "globalstrahlung_Wh_m2_sum",
                   "tage_seit_aussaat", "tageslicht_h","Tmean_gradC_sum",
                   "tageslicht_h_sum", "relLuftfeuchte_mean_prozent",
                   "Tmax_gradC","Tmin_gradC","daily_irrigation","daily_rain","niederschlag_mm_sum","Ton_prozent","Sand_prozent")

F_fine<- c("globalstrahlung_Wh_m2", "Tmean_gradC", "globalstrahlung_Wh_m2_sum",
           "tage_seit_aussaat", "tageslicht_h","Tmean_gradC_sum",
           "tageslicht_h_sum", "relLuftfeuchte_mean_prozent",
           "Tmax_gradC","Tmin_gradC","daily_irrigation","daily_rain","niederschlag_mm_sum","Ton_prozent","Sand_prozent","irmi", "ibi")

F_pre<- c( "Tmean_gradC",  "tageslicht_h","Tmean_gradC_sum",
           "tageslicht_h_sum", "relLuftfeuchte_mean_prozent",
           "Tmax_gradC","Tmin_gradC","daily_rain","niederschlag_mm_sum","Ton_prozent","Sand_prozent") 


setwd("C:/Users/jkangxxx/Desktop")
#setwd("C:/Users/Jongwon/Desktop")


train_df <- fread("MTL_correct_t9_train_df.csv") 
test_df <-fread("MTL_correct_t9_val_test_df.csv") 

n_past  <- 7
n_steps <- 35
time_steps <- n_past
lag_vars <- c("B0020_lag", "B2040_lag", "B4060_lag")
names_y <- c("B0020_nFK_prozent", "B2040_nFK_prozent",  "B4060_nFK_prozent")


rec <- recipe(train_df) %>% 
  step_mutate(
    B0020_lag = B0020_nFK_prozent,
    B2040_lag = B2040_nFK_prozent,
    B4060_lag = B4060_nFK_prozent,
  ) %>%
  update_role(all_of(names_y), new_role = "outcome") %>%
  update_role(-all_of(names_y), new_role = "predictor") %>%
  step_zv(all_predictors()) %>% 
  update_role(id_all, new_role = "id") %>%
  step_normalize(all_predictors()) %>%
  step_normalize(all_outcomes()) %>%
  prep()


train_baked_fine <- bake(rec, new_data = train_df) %>% select(all_of(c(F_fine_no_crop, "id_all", names_y, lag_vars)))


make_finetune_dataset <- function(baked_df, shuffle = FALSE) {
  local_feature_names <- setdiff(F_fine_no_crop, F_pre)
  weather_cols <- match(c(F_pre, lag_vars), names(baked_df))   
  local_cols   <- which(names(baked_df) %in% local_feature_names)
  target_cols  <- match(names_y, names(baked_df))
  
  map(unique(baked_df$id_all), ~{
    df_id <- baked_df[baked_df$id_all == .x, , drop = FALSE]
    
    x_weather <- head(as.matrix(df_id[, weather_cols]), -1)
    x_local   <- head(as.matrix(df_id[, local_cols]), -1)  # Now has 2 columns
    y         <- tail(as.matrix(df_id[, target_cols]), -7)
    
    # Weather Sequence (batch, 7, n_weather_features)
    ds_x_weather <- timeseries_dataset_from_array(
      data = x_weather, targets = NULL,
      sequence_length = 7, sampling_rate = 1, sequence_stride = 1,
      shuffle = shuffle, batch_size = 16L
    )
    
    # Local Sequence (batch, 7, 3) - NOW A TIME SERIES TOO
    ds_x_local <- timeseries_dataset_from_array(
      data = x_local, targets = NULL,
      sequence_length = 7, sampling_rate = 1, sequence_stride = 1,
      shuffle = shuffle, batch_size = 16L
    )
    
    
    ds_y1 <- tensor_slices_dataset(y[, 1, drop = FALSE]) %>% dataset_batch(16L)
    ds_y2 <- tensor_slices_dataset(y[, 2, drop = FALSE]) %>% dataset_batch(16L)
    ds_y3 <- tensor_slices_dataset(y[, 3, drop = FALSE]) %>% dataset_batch(16L)
    
    zip_datasets(list(
      zip_datasets(list(ds_x_weather, ds_x_local)), #inputs[[1]] and inputs[[2]] during fine-tuning
      zip_datasets(list(ds_y1, ds_y2, ds_y3))
    ))
  }) %>% reduce(dataset_concatenate)
}

fine_tune_ds <- make_finetune_dataset(train_baked_fine, shuffle = FALSE)


shared_model <- load_model("shared_model_checkpoint.keras") 
#shared_model_checkpoint.keras file saves the entire model (shared_model), which includes the pretrain_flatten and pretrain_head (dense) layers
encoder <- keras_model(inputs = shared_model$input, outputs = shared_model$get_layer("shared_dropout")$output)
# Creating encoder <- keras_model(...) simply exposes those already-trained layers as a standalone encoder model.
#encoder <- load_model("encoder_pretrained.keras")
encoder$trainable <- FALSE

stats <- tidy(rec, number = 4) %>%
  pivot_wider(names_from = statistic, values_from = value) 
# Extract means and standard deviations in order (Layer 1, Layer 2, Layer 3)
t_means <- c(
  stats$mean[1], 
  stats$mean[2], 
  stats$mean[3]
)

t_sds <- c(
  stats$sd[1], 
  stats$sd[2], 
  stats$sd[3]
)
MTL_LSTM_UWLF_tipping <- new_model_class(
  classname = "MTL_LSTM",
  
  # ── Define Parameters & Layers ──────────────────────────────────────────────
  initialize = function(n_tasks = 3, encoder_model = NULL, 
                        target_means = c(0, 0, 0), target_sds = c(1, 1, 1), ...) {
    super$initialize(...)
    self$n_tasks <- as.integer(n_tasks)
    self$encoder <- encoder_model   
    
    self$shared_lstm <- layer_lstm(units = 101, return_sequences = TRUE, name = "shared_lstm3") 
    
    self$dropout    <- layer_dropout(rate = 0.30, name = "context_dropout")
    self$local_lstm <- layer_lstm(units = 16, return_sequences = TRUE, name = "local_stream")
    
    # ── SHARED temporal attention ─────────────────────────────────────────────
    self$attn_dense1 <- layer_dense(units = 64, activation = "tanh") 
    self$attn_dense2 <- layer_dense(units = 1,  use_bias = FALSE)
    
    # ── Task heads (Cascaded structure) ───────────────────────────────────────
    self$head1_d1  <- layer_dense(units = 32, activation = "relu")
    self$head1_d2  <- layer_dense(units = 16, activation = "relu")
    self$head1_out <- layer_dense(units = 1,  name = "B0020")
    
    self$head2_d1      <- layer_dense(units = 32, activation = "relu")
    self$head2_d2      <- layer_dense(units = 16, activation = "relu")
    self$head2_raw     <- layer_dense(units = 1,  name = "B2040_raw")
    self$head2_merge_d <- layer_dense(units = 16, activation = "relu")
    self$head2_out     <- layer_dense(units = 1,  name = "B2040")
    
    self$head3_d1      <- layer_dense(units = 32, activation = "relu")
    self$head3_d2      <- layer_dense(units = 16, activation = "relu")
    self$head3_raw     <- layer_dense(units = 1,  name = "B4060_raw")
    self$head3_merge_d <- layer_dense(units = 16, activation = "relu")
    self$head3_out     <- layer_dense(units = 1,  name = "B4060")
    
    self$log_vars <- self$add_weight(
      name        = "log_vars",
      shape       = shape(n_tasks),
      initializer = "zeros",
      trainable   = TRUE
    )
    
    # ── Scaled Tipping-Bucket Field Capacity (100% nFK) ─────────────────────
    # Convert physical 100 nFK into the respective z-score scales for each layer
    fc_scaled_1 <- (100.0 - target_means[1]) / target_sds[1]
    fc_scaled_2 <- (100.0 - target_means[2]) / target_sds[2]
    
    self$fc1 <- op_convert_to_tensor(matrix(fc_scaled_1, nrow = 1), dtype = "float32")
    self$fc2 <- op_convert_to_tensor(matrix(fc_scaled_2, nrow = 1), dtype = "float32")
    
    init_raw <- log(exp(3.0) - 1)  # inverse-softplus(3.0)
    
    self$gate_sharpness1_raw <- self$add_weight(
      name        = "gate_sharpness1_raw",
      shape       = shape(1),
      initializer = initializer_constant(init_raw),
      trainable   = TRUE)
    
    self$gate_sharpness2_raw <- self$add_weight(
      name        = "gate_sharpness2_raw",
      shape       = shape(1),
      initializer = initializer_constant(init_raw),
      trainable   = TRUE)
  },
  
  # ── Feed Forward Logic ──────────────────────────────────────────────────────
  call = function(inputs, training = FALSE, ...) {
    weather_seq <- inputs[[1]] 
    local_seq   <- inputs[[2]] 
    
    encoded_weather <- self$encoder(weather_seq, training = FALSE)
    x_shared        <- self$shared_lstm(encoded_weather)
    x_local         <- self$local_lstm(local_seq)
    
    combined_seq <- layer_concatenate(list(x_shared, x_local), axis = -1L)
    
    e       <- combined_seq %>% self$attn_dense1() %>% self$attn_dense2()
    alpha   <- op_softmax(op_squeeze(e, axis = -1L), axis = -1L)
    context <- op_sum(combined_seq * op_expand_dims(alpha, axis = -1L), axis = 2L)
    
    context_full <- self$dropout(context, training = training)
    
    sharpness1 <- op_softplus(self$gate_sharpness1_raw[1])
    sharpness2 <- op_softplus(self$gate_sharpness2_raw[1])
    
    # ── Layer 1 (0-20cm) ─────────────────────────────────────────────────
    h1   <- context_full %>% self$head1_d1() %>% self$head1_d2()
    out1 <- self$head1_out(h1)
    
    # ── Tipping-bucket percolation: layer 1 -> layer 2 ──────────────────
    gate1   <- op_sigmoid(sharpness1 * (out1 - self$fc1))
    excess1 <- gate1 * (out1 - self$fc1)
    
    h2_raw <- context_full %>% self$head2_d1() %>% self$head2_d2() %>% self$head2_raw()
    out2   <- layer_concatenate(list(excess1, h2_raw)) %>%
      self$head2_merge_d() %>%
      self$head2_out()
    
    # ── Tipping-bucket percolation: layer 2 -> layer 3 ──────────────────
    gate2   <- op_sigmoid(sharpness2 * (out2 - self$fc2))
    excess2 <- gate2 * (out2 - self$fc2)
    
    h3_raw <- context_full %>% self$head3_d1() %>% self$head3_d2() %>% self$head3_raw()
    out3   <- layer_concatenate(list(excess2, h3_raw)) %>%
      self$head3_merge_d() %>%
      self$head3_out()
    
    list(out1, out2, out3)
  },
  
  # ── Custom Loss Implementation ──────────────────────────────────────────────
  compute_loss = function(x = NULL, y = NULL, y_pred = NULL, ...) { 
    total_loss <- 0.0
    for (t in 1:self$n_tasks) {
      y_true_t   <- op_cast(y[[t]], dtype = "float32")
      mse_t      <- op_mean(op_square(y_true_t - y_pred[[t]]))
      precision  <- op_exp(-self$log_vars[t])
      task_loss  <- 0.5 * precision * mse_t + 0.5 * self$log_vars[t]
      total_loss <- total_loss + task_loss
    }
    total_loss
  }
)

PenaltyTrackCallback <- new_callback_class(
  classname = "PenaltyTrackCallback",
  
  initialize = function(eval_inputs, ...) {
    super$initialize(...)
    self$history <- list()
    self$eval_inputs <- eval_inputs   # fixed batch of inputs for consistent cov_13 tracking
  },
  
  on_epoch_end = function(epoch, logs = NULL) {
    # 1. Penalty weight
 #   current_s <- as.numeric(self$model$log_penalty_w %>% as.array())
  #  penalty_w <- exp(current_s)
    
    # 2. cov_13 on a FIXED evaluation batch (consistent across epochs)
    preds <- self$model(self$eval_inputs, training = FALSE)
    p1 <- op_squeeze(preds[[1]], axis = -1L)
    p3 <- op_squeeze(preds[[3]], axis = -1L)
    p1_c <- p1 - op_mean(p1)
    p3_c <- p3 - op_mean(p3)
    cov_13 <- as.numeric(op_mean(p1_c * p3_c))
    
    self$history[[epoch + 1]] <- list(
      epoch     = epoch + 1,
 #     penalty_w = penalty_w,
      cov_13    = cov_13
    )
    
    cat(sprintf("Epoch %d |  cov_13 = %.6f\n",
                epoch + 1,  cov_13))
  }
)
eval_batch   <- fine_tune_ds %>% as_iterator() %>% iter_next()
eval_inputs  <- eval_batch[[1]]
penalty_cb <- PenaltyTrackCallback(eval_inputs = eval_inputs)



# Initialize the model, passing the scaling vectors
model <- MTL_LSTM_UWLF_tipping(
  n_tasks = 3, 
  encoder_model = encoder, 
  target_means = t_means, 
  target_sds = t_sds
)

lr_schedule <- function(epoch, lr) {
  if (epoch < 50) {
    return(0.001)
  } else {
    return(0.0001)
  }
}

lr_callback <- callback_learning_rate_scheduler(schedule = lr_schedule)

callbacks <- list(
  callback_model_checkpoint(filepath = "MTL_Geumuse.keras", monitor = "loss", save_best_only = TRUE),
  callback_early_stopping(monitor = "loss", patience = 25, min_delta = 0, restore_best_weights = TRUE),
  lr_callback, penalty_cb)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001), 
  metrics   = list(
    list(metric_mean_absolute_error(name = "B0020_mae")),
    list(metric_mean_absolute_error(name = "B2040_mae")),
    list(metric_mean_absolute_error(name = "B4060_mae"))
  )
)

history <- model %>% fit(x = fine_tune_ds, epochs = 100, shuffle = FALSE, callbacks = callbacks)

#_______________________GRAPH______________________________
track_df <- do.call(rbind, lapply(penalty_cb$history, as.data.frame))

trend_long <- track_df %>%
  pivot_longer(cols = cov_13, names_to = "Metric", values_to = "Value")

ggplot(trend_long, aes(x = epoch, y = Value, color=Metric)) +
  geom_line( linewidth = 1) +
  facet_wrap(~Metric, scales = "free_y", ncol = 1) +
  theme_minimal() +
  labs(title = "Penalty Weight and Covariance Over Training", x = "Epoch")


all_ids <- test_df %>% distinct(id_all) %>% pull(id_all)

stats <- tidy(rec, number = 4) %>%
  pivot_wider(names_from = statistic, values_from = value)

weather_vars <- c(F_pre, lag_vars)
local_vars   <- setdiff(F_fine_no_crop, F_pre)

all_results_df <- purrr::map_dfr(all_ids, function(id) {
  
  id_test <- test_df %>% filter(id_all == id)
  
  x_t <- bake(rec, id_test) %>%
    select(all_of(c(F_fine_no_crop, lag_vars))) %>%   
    mutate(across(everything(), as.numeric)) %>%  
    as.data.frame()
  
  weather_cols <- match(weather_vars, colnames(x_t))
  local_cols   <- match(local_vars, colnames(x_t))
  nn           <- match(lag_vars, colnames(x_t))
  
  preds <- matrix(NA, nrow = n_steps, ncol = length(names_y))
  
  for (i in seq_len(n_steps)) {
    X_weather <- array(
      as.matrix(x_t[(1:n_past) + i - 1, weather_cols]),
      dim = c(1, n_past, length(weather_cols))
    )
    
    X_local <- array(
      as.matrix(x_t[(1:n_past) + i - 1, local_cols]),
      dim = c(1, n_past, length(local_cols))
    )
    
    pred_list  <- model %>% predict(list(X_weather, X_local), verbose = 0)
    pred       <- sapply(pred_list, function(p) p[1, 1])
    preds[i, ] <- pred
    
    x_t[n_past + i, nn] <- pred
  }
  
  pred_df <- as.data.frame(preds) %>%
    setNames(names_y) %>%
    mutate(
      B0020_nFK_prozent = (B0020_nFK_prozent * stats$sd[1]) + stats$mean[1],
      B2040_nFK_prozent = (B2040_nFK_prozent * stats$sd[2]) + stats$mean[2],
      B4060_nFK_prozent = (B4060_nFK_prozent * stats$sd[3]) + stats$mean[3]
    )
  
  obs_df <- id_test[(n_past + 1):(n_past + n_steps), ] %>%
    select(all_of(names_y))
  
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

#file.rename("MTL_Geumuse.keras", "Tipping_0721_no_crop_v1.keras")
