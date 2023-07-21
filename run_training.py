import os
import torch
import sys
import torch_geometric
from Dataset import PocketDataset
import GNN_model as model
import numpy as np

# Create instances of data
train_dataset = PocketDataset(root = 'training_data')
# Create Dataloader 
data_train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=10)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create instance of the GNN 
model=model.Pocket_GNN()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

#Compute mean absolute error
def weighted_mae_loss(y_pred, y_true, weight_factor, gt_std, gt_mean):
    absolute_errors = torch.abs(y_pred - y_true)
    weights = 1 + weight_factor * (y_true*gt_std+gt_mean)
    weighted_errors = weights * absolute_errors
    loss = torch.sum(weighted_errors) / torch.sum(weights)
    return loss

# Compute train loss function
def train():
	model.train()
	loss_sum = 0
	total_size = 0
	for batch in data_train_loader:
		batch=batch.to(device)
		optimizer.zero_grad()
		batch_size = len(batch.y)
		pred_y = model(batch)
		loss=weighted_mae_loss(pred_y.squeeze(), batch.y.squeeze(), 4, 0.14342581912400262 , 0.1849439336325112)
		loss.backward()
		loss_sum+= batch_size*loss.item()
		total_size+= batch_size
		optimizer.step()
	return loss_sum/total_size
loss = train()

number_of_epochs=int(sys.argv[2])
loss_file=sys.argv[3]
output_directory=sys.argv[4]
if( not os.path.exists(output_directory)):
	os.makedirs(output_directory)

# Training loop

saving_period=1;
training_loss=[]
if  os.path.exists(loss_file):
	training_loss = np.loadtxt(loss_file)
	final_index = len(training_loss)
	model_path = os.path.join(output_directory, 'epoch'+str(final_index)+'.pth')
	start_epoch = final_index+1
	model = torch.load(model_path)
else:
	model = model
	start_epoch = 1
# Load the model 
model = model.to(device)
end_epoch = start_epoch + number_of_epochs
file = open(loss_file, 'w')
for epoch in range(start_epoch, end_epoch):
	train_loss = train()
	training_loss = np.append(training_loss, train_loss)
	file.write('epoch ='+ str(epoch)+ ', train mae ='+ str(train_loss)+'\n')
	print('epoch =', epoch, ', train mae =', train_loss)
	# save models
	if (epoch==0) or (epoch%saving_period==0) or (epoch==number_of_epochs):
		torch.save(model, output_directory+'/epoch'+str(epoch)+'.pth')
#np.savetxt(loss_file, training_loss, delimiter= '\n')
file.close()