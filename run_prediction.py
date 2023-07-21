import os
import sys
import torch
from numpy import savetxt
import Dataset


dataset_name = sys.argv[2]
model_name = sys.argv[3]
graph_name = sys.argv[4:]
model_file = os.path.join('./output_grained_trained_models', model_name)
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

def pred_train():
    for graph in graph_name:
        data = Dataset.read_graph(graph)
        data = data.to(device)
        # create directory 
        model = torch.load(model_file)
        model = model.to(device)
        model.train()
        pred_y = model(data)
        path = os.path.join('./output_predicted_values/train_set', model_name, graph)
        savetxt(path, pred_y.squeeze().detach().numpy())

pred_train()
