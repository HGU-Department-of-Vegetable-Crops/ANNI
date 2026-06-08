#Bayesian optimization (for hyperparmeters)

library(rBayesianOptimization)

train_mtl <- function(log_lr,dropout) {
  
  #lstm1_units <- as.integer(round(lstm1_units))
  #lstm2_units <- as.integer(round(lstm2_units))
  lr          <- 10^log_lr
  
  # Rebuild model class with tunable params
  MTL_LSTM_UWFL_BO <- new_model_class(
    classname = "MTL_LSTM_UWFL_BO",
    
    initialize = function(n_tasks = 3, ...) {
      super$initialize(...)
      self$n_tasks <- as.integer(n_tasks)
      
      # Shared encoder with tunable units
      self$lstm1   <- layer_lstm(units = 128, return_sequences = TRUE)
      self$lstm2   <- layer_lstm(units = 64, return_sequences = TRUE)
      self$dropout <- layer_dropout(rate = dropout)
      
      self$attn_dense1 <- layer_dense(units = 128, activation = "tanh") # Wh * hk + bh
      self$attn_dense2 <- layer_dense(units = 1, use_bias = FALSE)       # u^T
      
      # 3 separate heads (fixed architecture)
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
      
      # UWFL learnable log-variance per task
      self$log_vars <- self$add_weight(
        name        = "log_vars",
        shape       = shape(n_tasks),
        initializer = "zeros",
        trainable   = TRUE
      )
    },
    
    call = function(inputs, training = FALSE, ...) {
      # ── Shared Encoder ──────────────────────────────────────────────
      x <- inputs %>%
        self$lstm1(training = training) %>%   # (batch, 7, 223)
        self$lstm2(training = training)        # (batch, 7, 101)
      
      # ── Attention Mechanism ─────────────────────────────────────────
      # Step 1: eₖ = u^T * tanh(Wh * hk + bh)
      e <- x %>%
        self$attn_dense1() %>%    # tanh(Wh * hk + bh) → (batch, 7, 101)
        self$attn_dense2()         # u^T * (...)         → (batch, 7, 1)
      
      # Step 2: squeeze + softmax → αᵢ
      e_squeezed <- op_squeeze(e, axis = -1L)        # (batch, 7)
      alpha       <- op_softmax(e_squeezed, axis = -1L)  # (batch, 7)
      
      # Step 3: expand α for weighted multiply
      alpha_expanded <- op_expand_dims(alpha, axis = -1L)  # (batch, 7, 1)
      
      # Step 4: weighted sum → context vector
      context <- op_sum(x * alpha_expanded, axis = 2L)     # (batch, 101)
      
      # ── Dropout ─────────────────────────────────────────────────────
      context <- self$dropout(context, training = training)
      
      out1 <- self$head1(context, training = training)
      out2 <- self$head2(context, training = training)
      out3 <- self$head3(context, training = training)
      
      list(out1, out2, out3)
    },
    
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
  
  # Build & compile
  model <- MTL_LSTM_UWFL_BO(n_tasks = 3)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    metrics   = list(
      list(metric_mean_absolute_error(name = "B0020_mae")),
      list(metric_mean_absolute_error(name = "B2040_mae")),
      list(metric_mean_absolute_error(name = "B4060_mae"))
    )
  )
  
  history <- model %>% fit(
    final_train,
    epochs          = 500,          
    validation_data = final_val,
    shuffle         = FALSE,
    verbose         = 0,
    callbacks       = list(
      callback_early_stopping(
        monitor              = "val_loss",
        patience             = 10,
        min_delta            = 0.001,
        restore_best_weights = TRUE
      ),
      callback_reduce_lr_on_plateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 5
      )
    )
    # NOTE: no model_checkpoint during BO search — saves time
  )
  
  best_val_loss <- min(history$metrics$val_loss)
  list(Score = -best_val_loss, Pred = 0)
}

# --- Bounds ---
bounds <- list(
 # lstm1_units = c(128L,  256L),   
 # lstm2_units = c(64L,  128L),  
  dropout     = c(0.1,  0.5),
  log_lr      = c(-5,   -3)     
)

# --- Run BO ---
set.seed(42)
bo <- BayesianOptimization(
  FUN          = train_lstm,
  bounds       = bounds,
  init_points  = 8,    # random exploration first
  n_iter       = 20,   # BO iterations after
  acq          = "ucb",
  kappa        = 2.576 # exploration/exploitation tradeoff
)

# --- Results ---
bo$Best_Par
best_lr <- 10^bo$Best_Par["log_lr"]
cat("Best lr:", best_lr, "\n")
cat("Best lstm1_units:", round(bo$Best_Par["lstm1_units"]), "\n")
cat("Best lstm2_units:", round(bo$Best_Par["lstm2_units"]), "\n")
cat("Best dropout:",     bo$Best_Par["dropout"], "\n")

