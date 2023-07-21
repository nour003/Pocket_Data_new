import torch
import Dataset 
import sys

dataset_name=sys.argv[2]
model_files=sys.argv[3:]
dataset = Dataset.PocketDataset(root = dataset_name)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Compute mean absolute error
def weighted_mae_loss(y_pred, y_true, weight_factor, gt_std, gt_mean):
    absolute_errors = torch.abs(y_pred - y_true)
    weights = 1 + weight_factor * (y_true*gt_std+gt_mean) 
    weighted_errors = weights * absolute_errors
    loss = torch.mean(weighted_errors)
    return loss

# Compute validation loss
@torch.no_grad()
def calc_mse(data):
	model.eval()
	data=data.to(device)
	pred_y=model(data)
	loss=weighted_mae_loss(pred_y.squeeze(), data.y.squeeze(), 4, 0.14342581912400262 , 0.1849439336325112)
	return loss.item()

for model_file in (model_files):
	model = torch.load(model_file)
	model = model.to(device)
	mae_sum = 0
	total_size = 0
	for data in dataset:
		mae_sum+= len(data.x)*calc_mse(data)
		total_size+= len(data.x)
	print(mae_sum/total_size)
