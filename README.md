"# Pocket_data_new" 
Here we used dropout=0.25 , learning_rate= 0.001 and loss_weight_factor=8 and chnaged model configuration( removed two layers)
minimum_validation_loss=  0.7280477426988972

# Pocket_Data_new

# train GNN
./run_training.bash

# test saved trained GNN models
./run_evaluation.bash

# plot train and validation loss curves
./plot_loss.bash

# compute ground truth predictions for training and validation dataset
./run_prediction.bash 