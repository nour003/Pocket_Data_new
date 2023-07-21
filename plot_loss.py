import matplotlib.pyplot as plt
from utils import read_floats
import sys
dataset_name=sys.argv[2]
train_file=sys.argv[3]
validation_file=sys.argv[4]
#plot the MSE of training data
def plot_mse():
	mse_train = read_floats(train_file)
	mse_vald = read_floats(validation_file)
	min_index= mse_vald.index(min(mse_vald))
	plt.plot(mse_vald, 'r', label = 'vald loss using ' + dataset_name)
	plt.plot(mse_train, 'b', label = 'train loss')
	plt.scatter(min_index, min(mse_vald), color='r')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.savefig("loss_using_"+dataset_name)
plot_mse()
