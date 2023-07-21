#import libraires
import pandas
import torch
import torch_geometric
import sys
import utils
import os

#global var
ROOT_PATH = sys.argv[1]
PATH = os.path.join(ROOT_PATH, 'prepared_input_data')
#################################
# Make a graph dataset
def read_graph(file_name):
    # Get dataframes
    path = os.path.join(PATH, file_name, file_name)
    df_links = pandas.read_csv(path+'_links.csv')
    df_nodes = pandas.read_csv(path+'_nodes.csv')
    # Create tensors 
    x=torch.tensor(df_nodes[['radius', 'voromqa_sas_potential','residue_mean_sas_potential', 'residue_sum_sas_potential','residue_size','sas_area', 'voromqa_sas_energy','voromqa_depth','voromqa_score_a', 'voromqa_score_r', 'volume', \
                    'volume_vdw', 'ufsr_a1', 'ufsr_a2', 'ufsr_a3', 'ufsr_b1','ufsr_b2', 'ufsr_b3', 'ufsr_c1', 'ufsr_c2', 'ufsr_c3', 'ev28', 'ev56']].values, dtype=torch.float32)
    y=torch.tensor(df_nodes[['ground_truth']].values, dtype=torch.float32)
    edge_index = torch.tensor(df_links[['atom_index1', 'atom_index2']].values.T, dtype=torch.long)
    edge_attr = torch.tensor(df_links[['area', 'boundary', 'distance', 'voromqa_energy', 'seq_sep_class','covalent_bond', 'hbond']].values, dtype=torch.float32)
    # Create data graph
    graph=torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return graph
################################
#creating dataset class
class PocketDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, transform = None, pre_transform=None, pre_filter=None, log=False):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])     
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        raw_prefixes= utils.read_strings(ROOT_PATH+'/'+self.root+'_files_prefixes')
        data_list=[read_graph(raw_prefix) for raw_prefix in raw_prefixes]
        if self.pre_filter is not None:
            data_list=[data for data in data_list if self.pre_filter(data)]
	
        if self.pre_transform is not None:
            data_list=[self.pre_transform(data) for data in data_list]
	
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
