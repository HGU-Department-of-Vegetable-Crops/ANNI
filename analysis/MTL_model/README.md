# Multi-task learning (MTL) model

To improve the previous models (LSTM and MLP) and to simplify the apporach by avoiding the Ensemble method, a multi-task learning model was prosposed.
The model consists of a shared LSTM-based layer for temporal feature extraction and task-specific multilayer perceptron (MLP) layers for predicting SM at three depths (0.2, 0.4, and 0.6 m), reflecting both the inherent dependence of SM across depths and their partial independent behaviours. An attention mechanism is incorporated to bridge shared and task-specific representations.

Data batches (IDs) were created by aggregating the dataset according to location, replication, treatment, and sowing date, resulting in a total of 72 IDs. To prevent overfitting, the dataset was partitioned into training (75 % of IDs), validation (15 %), and test (10 %) sets. 
