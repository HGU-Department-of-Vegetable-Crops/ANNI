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
#library(ggpmisc)

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



names_y <- c(
  "B0020_nFK_prozent", 
  "B2040_nFK_prozent",
  "B4060_nFK_prozent")


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
test_baked_fine <- bake(rec, new_data = test_df) %>% select(all_of(c(F_fine_no_crop, "id_all", names_y, lag_vars)))


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
fine_tune_ds_test <- make_finetune_dataset(test_baked_fine, shuffle = FALSE)

MTL_LSTM_physics_cascade <- new_model_class(
  classname = "MTL_LSTM",
  
  # ── Define Parameters & Layers ──────────────────────────────────────────────
  initialize = function(n_tasks = 3, encoder_model = NULL, ...) {
    super$initialize(...)
    self$n_tasks <- as.integer(n_tasks)
    self$encoder <- encoder_model   
    self$shared_lstm<- layer_lstm(units = 101, return_sequences = TRUE, name = "shared_lstm3") 
    
    self$dropout     <- layer_dropout(rate = 0.30, name = "context_dropout")
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
    
    self$log_penalty_w <- self$add_weight(
      name        = "log_penalty_w",
      shape       = shape(1),
      initializer = initializer_constant(log(0.1)),
      trainable   = TRUE)
    
    
  },
  
  # ── Feed Forward Logic ──────────────────────────────────────────────────────
  call = function(inputs, training = FALSE, ...) {
    weather_seq <- inputs[[1]]   # (batch, time_steps, n_weather_features)
    local_seq  <- inputs[[2]]   # (batch, 3) 
    
    # Forward pass through the frozen pre-trained encoder
    encoded_weather <- self$encoder(weather_seq, training = FALSE)
    x_shared        <- self$shared_lstm(encoded_weather) #  (batch, 7, 101)
    
    x_local  <- self$local_lstm(local_seq)                  #  (batch, 7, 16)
    
    # 2. Sequence Concatenation 
    # (batch, 7, 117)
    combined_seq <- layer_concatenate(list(x_shared, x_local), axis = -1L)
    
    # 3.  Temporal Attention mechanism
    # The attention layer now decides which days are important based on BOTH 
    e       <- combined_seq %>% self$attn_dense1() %>% self$attn_dense2()
    alpha   <- op_softmax(op_squeeze(e, axis = -1L), axis = -1L)
    context <- op_sum(combined_seq * op_expand_dims(alpha, axis = -1L), axis = 2L)
    
    # The resulting context vector is a highly informed (batch, 117) representation
    context_full <- self$dropout(context, training = training)
    
    # Cascade Task Predictions
    h1   <- context_full %>% self$head1_d1() %>% self$head1_d2()
    out1 <- self$head1_out(h1)
    
    h2_raw <- context_full %>% self$head2_d1() %>% self$head2_d2() %>% self$head2_raw()
    out2   <- layer_concatenate(list(out1, h2_raw)) %>% self$head2_merge_d() %>% self$head2_out()
    
    h3_raw <- context_full %>% self$head3_d1() %>% self$head3_d2() %>% self$head3_raw()
    out3 <- layer_concatenate(list(out2, h3_raw)) %>% self$head3_merge_d() %>% self$head3_out()
    
    
    list(out1, out2, out3)
  },
  
  # ── Custom Loss Implementation ──────────────────────────────────────────────
  compute_loss = function(x = NULL, y = NULL, y_pred = NULL, ...) {
    total_loss <- 0.0
    for (t in 1:self$n_tasks) {
      y_true_t   <- op_cast(y[[t]], dtype = "float32")
      mse_t      <- op_mean(op_square(y_true_t - y_pred[[t]]))
      total_loss <- total_loss + mse_t
    }
    
    p1 <- op_squeeze(y_pred[[1]], axis = -1L)
    p2 <- op_squeeze(y_pred[[2]], axis = -1L)
    p3 <- op_squeeze(y_pred[[3]], axis = -1L)
    
    p1_c <- p1 - op_mean(p1)
    p2_c <- p2 - op_mean(p2)
    p3_c <- p3 - op_mean(p3)
    
    cov_13 <- op_mean(p1_c * p3_c)
    cov_penalty <- op_relu(cov_13)  # cov_13  #op_relu(cov_13)
    
    penalty_w  <- op_exp(self$log_penalty_w[1])
    total_loss + (penalty_w * cov_penalty)
  }
)

shared_model <- load_model("shared_model_checkpoint.keras")
encoder <- keras_model(inputs = shared_model$input, outputs = shared_model$get_layer("shared_dropout")$output)
encoder$trainable <- FALSE

# ── 2. INSTANTIATE AND LOAD TRAINED WEIGHTS ──────────────────────────────────
# Create the structural shell
physics_model <- MTL_LSTM_physics_cascade(n_tasks = 3, encoder_model = encoder)
weather_vars <- c(F_pre, lag_vars)
local_vars   <- setdiff(F_fine_no_crop, F_pre)
dummy_weather <- array(0, dim = c(1, n_past, length(weather_vars)))
dummy_local   <- array(0, dim = c(1, n_past, length(local_vars)))

# Call once to build internal Keras states
physics_model(list(dummy_weather, dummy_local))
physics_model %>% load_model_weights("Transfer_0708_no_crop_relu_v7.keras")


all_ids <- test_df %>% distinct(id_all) %>% pull(id_all)

stats <- tidy(rec, number = 4) %>%
  pivot_wider(names_from = statistic, values_from = value)



relu_results_df <- purrr::map_dfr(all_ids, function(id) {
  
  id_test <- test_df %>% filter(id_all == id)
  
  x_t <- bake(rec, id_test) %>%
    select(all_of(c(F_fine, lag_vars))) %>%   
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
    
    pred_list  <- physics_model %>% predict(list(X_weather, X_local), verbose = 0)
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
relu_plot_df_long <- relu_results_df %>%
  pivot_longer(
    -c(step, id_all),
    names_to      = c("variable", "type"),
    names_pattern = "(.+)_(pred|obs)"
  )



# 4. Calculate metrics smoothly using the pivoted dataframe
all_metrics <- relu_plot_df_long %>%
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


threshold_ggplot <- relu_plot_df_long %>% 
  pivot_wider(names_from = type, values_from = value) %>%
 # filter(obs < 60) %>% 
  ggplot(aes(x = obs, y = pred)) +
  geom_point(alpha = 0.4, color = "midnightblue") + 
  
  # 2. Reference 1:1 line
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "firebrick", linewidth = 0.7) +
  
  # 3. Dynamic R² annotation from ggpmisc
  stat_poly_eq(
    aes(label = after_stat(rr.label)),
    formula = y ~ x,
    parse   = TRUE,
    label.x = 0.05, # Use relative coordinates so they align nicely near the top-left
    label.y = 0.95,
    size    = 4
  ) +
  facet_wrap(~ variable) + #, scales = "free"
  labs(
    title = "Predicted vs. Observed Soil Moisture during Dry Conditions",
    subtitle = "Filtered for Observed Soil Moisture < 60% AWC (",
    x = "Observed (% AWC)", 
    y = "Predicted (% AWC)"
  ) +
  
  # 4. Restructure limits to zoom into the 0-60% target window
 # scale_x_continuous(breaks = seq(0, 60, 10), limits = c(0, 65)) +
 # scale_y_continuous(breaks = seq(0, 60, 10), limits = c(0, 65)) +
  
  # 5. Crucial for 1:1 plots: forces a true geometric square aspect ratio
  coord_fixed() + 
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 11) # Cleans up facet titles (B0020, etc.)
  )

threshold_ggplot




relu_random_plot<-relu_plot_df_long %>% filter(id_all %in% c(1,4,21, 25,39,75)) 
#c(1, 13,25,44,74)
#c(1,6,44,70,74) -> DWD

ggplot(relu_random_plot, aes(x = step, y = value, colour = type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2)+
  facet_grid(variable ~ id_all) +
  scale_colour_manual(
    values = c(obs = "#2C3E50", pred = "#E74C3C"),
    labels = c(obs = "Observed", pred = "Predicted")
  ) +
  scale_y_continuous(limits = c(0, 200), breaks = seq(0, 200, 50)) +
  labs(
    title    = "Predicted vs Observed — All test IDs",
    x        = "7 Days after planting",
    y        = "AWC",
    colour   = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    strip.text.y    = element_text(angle = 0, size = 9),
    strip.text.x    = element_text(size = 8)
  )

ggplot(relu_plot_df_long, aes(x = step, y = value, colour = type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2) +
  facet_grid(variable ~ id_all) +
  scale_colour_manual(
    values = c(obs = "#2C3E50", pred = "#E74C3C"),
    labels = c(obs = "Observed", pred = "Predicted")
  ) +
  scale_y_continuous(limits = c(0, 200), breaks = seq(0, 200, 50)) +
  labs(
    title    = "Predicted vs Observed — All test IDs",
    x        = "7 Days after planting",
    y        = "AWC",
    colour   = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    strip.text.y    = element_text(angle = 0, size = 9),
    strip.text.x    = element_text(size = 8)
  )





#================DWD==============================___
dwd_df <-fread("DWD_df_all.csv") 

all_DWD_ids <- dwd_df %>% distinct(id_all) %>% pull(id_all)


all_DWD_df <- purrr::map_dfr(all_DWD_ids, function(id) {
  
  id_test <- dwd_df %>% filter(id_all == id)
  
  x_t <- bake(rec, id_test) %>%
    select(all_of(c(F_fine, lag_vars))) %>%   
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
    
    pred_list  <- physics_model %>% predict(list(X_weather, X_local), verbose = 0)
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
    pred_df %>% rename_with(~ paste0(.x, "_dwd_pred")),
    obs_df  %>% rename_with(~ paste0(.x, "_dwd_obs"))
  ) %>%
    mutate(step = row_number(), id_all = as.character(id))
})




# 3. Pivot the data long for plotting
DWD_plot_df_long <- all_DWD_df %>%
  pivot_longer(
    -c(step, id_all),
    names_to      = c("variable", "type"),
    names_pattern = "(.+)_(dwd_pred|dwd_obs)"
  ) %>%
  mutate(type = recode(type, dwd_pred = "dwd", dwd_obs = "obs"))

DWD_metric_df_long <- all_DWD_df %>%
  pivot_longer(
    -c(step, id_all),
    names_to      = c("variable", "type"),
    names_pattern = "(.+)_(pred|obs)"
  )


# 4. Calculate metrics smoothly using the pivoted dataframe
DWD_all_metrics <- DWD_metric_df_long %>%
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

DWD_avg_metrics <- DWD_all_metrics %>%
  group_by(variable) %>%
  summarise(
    RMSE    = mean(RMSE,    na.rm = TRUE),
    nRMSE   = mean(nRMSE,   na.rm = TRUE),
    d_index = mean(d_index, na.rm = TRUE),
    n_ids   = n()
  )

DWD_avg_metrics

DWD_true_plot_df_long <- bind_rows(
  DWD_plot_df_long %>% filter(type == "dwd"),                     
  relu_plot_df_long %>% filter(id_all %in% all_DWD_ids)                
)


#DWD_random_plot <-DWD_true_plot_df_long %>% filter(!id_all%in%c(47,10,39,70,74))

DWD_low_plot <-DWD_true_plot_df_long %>% filter(id_all%in%c(39))
DWD_high_plot1 <-DWD_true_plot_df_long %>% filter(id_all%in%c(44))
DWD_high_plot2 <-DWD_true_plot_df_long %>% filter(id_all%in%c(75))
ggplot(DWD_high_plot2, aes(x = step, y = value, colour = type, linetype = type, linewidth = type)) +
  geom_line() +
  geom_point(size = 0.8) +
  facet_grid(variable ~ id_all) +
  scale_colour_manual(
    values = c(obs = "#2C3E50", pred = "#E74C3C", dwd = "#27AE60"),
    labels = c(obs = "Observed", pred = "Predicted", dwd = "DWD Predicted")
  ) +
  scale_linetype_manual(
    values = c(obs = "dashed", pred = "solid", dwd = "solid"),
    labels = c(obs = "Observed", pred = "Predicted", dwd = "DWD Predicted")
  ) +
  scale_linewidth_manual(
    values = c(obs = 0.8, pred = 0.8, dwd = 1.2),
    guide = "none" # Removes the size legend since it's redundant
  ) +
  scale_y_continuous(limits = c(0, 200), breaks = seq(0, 200, 50)) +
  labs(
    title    = "Predicted vs Observed vs DWD — All test IDs",
    x        = "7 Days after planting",
    y        = "AWC",
    colour   = NULL,
    linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    strip.text.y    = element_text(angle = 0, size = 9),
    strip.text.x    = element_text(size = 8)
  )


#===============Sensitivity===========

deltas <- seq(-30, 30, by = 5)
deltas <- deltas[deltas != 0]

# ── 1. Perturbation function (unchanged) ─────────────────────────
apply_perturbation <- function(df, cols, delta) {
  df %>%
    group_by(id_all) %>%
    mutate(
      day_index = row_number(),
      across(
        all_of(cols),
        ~ if_else(day_index <= n_past, .x + delta, .x)   
      )
    ) %>%
    select(-day_index) %>%
    ungroup()
}

# ── 3. Main Sensitivity Loop ─────────────────────────────────────
sensitivity_results <- purrr::map_dfr(deltas, function(delta_val) {
  
  cat(sprintf("\nRunning sensitivity for delta: %+.0f%%\n", delta_val))
  
  # Step A: Perturb raw test data
  perturbed_data <- apply_perturbation(test_df, names_y, delta_val)
  
  # Step B: Bake (apply recipe transformations)
  baked_data <- bake(rec, perturbed_data)
  
  # Step C: Per id_all prediction loop
  purrr::map_dfr(unique(baked_data$id_all), function(id_val) {
    
    id_data <- baked_data %>%
      filter(id_all == id_val) %>%
      select(all_of(c(F_fine, lag_vars))) %>%
      mutate(across(everything(), as.numeric)) %>%
      as.data.frame()
    
    n_steps <- nrow(id_data) - n_past
    if (n_steps <= 0) return(NULL)
    
    # Map column indices for the split tensors
    weather_cols <- match(weather_vars, colnames(id_data))
    local_cols   <- match(local_vars, colnames(id_data))
    
    x_t <- id_data   # mutable copy
    preds <- matrix(NA, nrow = n_steps, ncol = length(names_y))
    
    for (i in seq_len(n_steps)) {
      
      # Build the Dual-Tensors
      X_weather <- array(
        as.matrix(x_t[(1:n_past) + i - 1, weather_cols]),
        dim = c(1, n_past, length(weather_cols))
      )
      
      X_local <- array(
        as.matrix(x_t[(1:n_past) + i - 1, local_cols]),
        dim = c(1, n_past, length(local_cols))
      )
      
      # Predict using the multi-input list
      pred_list  <- physics_model %>% predict(list(X_weather, X_local), verbose = 0)
      pred       <- sapply(pred_list, function(p) p[1, 1])
      preds[i, ] <- pred
      
      # Feed predictions back safely by exact name
      names(pred) <- lag_vars
      if ((i + n_past) <= nrow(x_t)) {
        x_t[i + n_past, lag_vars] <- pred
      }
    }
    
    # Step D: Denormalise predictions using string mapping
    pred_df <- as.data.frame(preds) %>%
      setNames(names_y) %>%
      mutate(
        B0020_nFK_prozent = (B0020_nFK_prozent * stats$sd[1]) + stats$mean[1],
        B2040_nFK_prozent = (B2040_nFK_prozent * stats$sd[2]) + stats$mean[2],
        B4060_nFK_prozent = (B4060_nFK_prozent * stats$sd[3]) + stats$mean[3]
      )
    
    # Step E: Get observed values for same window
    obs_raw <- test_df %>%
      filter(id_all == id_val) %>%
      slice((n_past + 1):(n_past + n_steps)) %>%
      select(all_of(names_y))
    
    # Step F: Bind and return
    bind_cols(
      pred_df %>% rename_with(~ paste0(.x, "_pred")),
      obs_raw  %>% rename_with(~ paste0(.x, "_obs"))
    ) %>%
      mutate(
        step         = row_number(),
        id_all       = as.character(id_val),
        perturbation = delta_val    
      )
  })
})


# 12*25*35
dim(sensitivity_results)

head(sensitivity_results)

all_metrics <- sensitivity_results %>%
  pivot_longer(
    -c(step, id_all, perturbation),
    names_to      = c("variable", "type"),
    names_pattern = "(.+)_(pred|obs)"
  ) %>%
  pivot_wider(names_from = type, values_from = value) %>%
  filter(!is.na(obs) & !is.na(pred)) %>%
  group_by(perturbation, id_all, variable) %>%
  summarise(
    RMSE    = rmse_vec(truth = obs, estimate = pred),
    mean_obs = mean(obs),
    d_index  = 1 - sum((pred - obs)^2) /
      sum((abs(pred - mean_obs) + abs(obs - mean_obs))^2),
    .groups = "drop"
  ) %>%
  select(-mean_obs)

# ── Average across id_all → one row per perturbation × depth ─────
avg_metrics <- all_metrics %>%
  group_by(perturbation, variable) %>%
  summarise(
    RMSE    = mean(RMSE,    na.rm = TRUE),
    d_index = mean(d_index, na.rm = TRUE),
    n_ids   = n(),
    .groups = "drop"
  ) %>%
  mutate(variable = factor(variable,
                           levels = c("B0020_nFK_prozent",
                                      "B2040_nFK_prozent",
                                      "B4060_nFK_prozent"),
                           labels = c("B0020 (0–20cm)",
                                      "B2040 (20–40cm)",
                                      "B4060 (40–60cm)")))

avg_metrics



ggplot(avg_metrics,
       aes(x = perturbation, y = RMSE,
           color = variable, group = variable)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("#2166ac", "#4dac26", "#b2182b")) +
  labs(title = "RMSE by Perturbation",
       x = "SM Perturbation (%)", y = "RMSE",
       color = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom")

ggplot(avg_metrics,
       aes(x = perturbation, y = d_index,
           color = variable, group = variable)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("#2166ac", "#4dac26", "#b2182b")) +
  labs(title = "Index of Agreement (d) by Perturbation",
       x = "SM Perturbation (%)", y = "d index",
       color = NULL) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom")
