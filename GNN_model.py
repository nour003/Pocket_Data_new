import torch
import torch_geometric

class Pocket_GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linA=torch.nn.Linear(23, 23)
        self.linB=torch.nn.Linear(23, 23)
        self.conv1=torch_geometric.nn.TransformerConv(23, 24, heads=8, edge_dim=7, dropout=0.25)
        self.conv4=torch_geometric.nn.TransformerConv(24*8, 8, heads=8, edge_dim=7, dropout=0.25)
        self.lin1=torch.nn.Linear(8*8, 20)
        self.lin2=torch.nn.Linear(20, 1)
    def forward(self, data):
        x=data.x
        x=self.linA(x)
        x=torch.nn.functional.elu(x)
        x=self.linB(x)
        x=torch.nn.functional.elu(x)
        x=self.conv1(x, data.edge_index, data.edge_attr)
        x=torch.nn.functional.elu(x)
        x=self.conv4(x, data.edge_index, data.edge_attr)
        x=torch.nn.functional.elu(x)
        x=self.lin1(x)
        x=torch.nn.functional.elu(x)
        return self.lin2(x)
    