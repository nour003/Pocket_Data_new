import pandas
import os
import utils
import sys

PATH = sys.argv[1]
PATH_ROOT = os.path.join(PATH,'separated_graphs')

def normalize_nodes(file_name):    
    # create dataframes 
    node_path = os.path.join(PATH_ROOT, file_name, file_name+'_nodes.csv')
    df_nodes = pandas.read_csv(node_path)
    # Normalize nodes
    nodes_to_normalize = ['radius', 'voromqa_sas_potential','residue_mean_sas_potential', 'residue_sum_sas_potential','residue_size','sas_area', 'voromqa_sas_energy','voromqa_depth','voromqa_score_a', 'voromqa_score_r', 'volume', \
                    'volume_vdw', 'ufsr_a1', 'ufsr_a2', 'ufsr_a3', 'ufsr_b1','ufsr_b2', 'ufsr_b3', 'ufsr_c1', 'ufsr_c2', 'ufsr_c3', 'ev28', 'ev56','ground_truth']
    node_stds = utils.read_floats(os.path.join(PATH, 'train_data', 'stds', 'nodes_std.txt'))
    node_means = utils.read_floats(os.path.join(PATH, 'train_data', 'means', 'nodes_mean.txt'))
    # Exclude columns with standard deviation of 0 from normalization
    for index, column in enumerate(nodes_to_normalize):
        df_nodes[column].values[:] = (df_nodes[column].values[:] - node_means[index]) / node_stds[index]
   
    # Save the normalized DataFrame as a new CSV file
    path = os.path.join(PATH, "prepared_input_data", file_name)
    df_nodes.to_csv(os.path.join(path,file_name+"_nodes.csv"), index=False)
    return df_nodes

def normalize_links(file_name):
    link_path = os.path.join(PATH_ROOT, file_name, file_name+'_links.csv')
    df_links = pandas.read_csv(link_path)
     # Normalize links
    columns_to_normalize = ['area', 'boundary', 'distance', 'voromqa_energy', 'seq_sep_class','covalent_bond', 'hbond']
    link_stds = utils.read_floats(os.path.join(PATH, 'train_data', 'stds', 'links_std.txt'))
    link_means = utils.read_floats(os.path.join(PATH, 'train_data', 'means', 'links_mean.txt'))
    # Exclude columns with standard deviation of 0 from normalization
    columns_to_normalize = [col for col in columns_to_normalize if link_stds[columns_to_normalize.index(col)] != 0]
    for index, column in enumerate(columns_to_normalize):
        df_links[column].values[:] = (df_links[column].values[:] - link_means[index]) / link_stds[index]
    # Add 'is_self' attribute 
    df_links['is_self'] = len(df_links.index) * [0]
    # Save the normalized DataFrame as a new CSV file
    return df_links

# operatins only on link_files
##################################
# Add bidirectional links
def add_bidirectional_links(file_name):    
    df_links= normalize_links(file_name)
    #invert atom index columns
    df_bi = df_links.copy()
    columns_to_replace_1 = ['atom_index1','ID1_chainID', 'ID1_resSeq', 'ID1_iCode', 'ID1_serial','ID1_altLoc', 'ID1_resName', 'ID1_name']
    columns_to_replace_2 =['atom_index2','ID2_chainID', 'ID2_resSeq', 'ID2_iCode', 'ID2_serial','ID2_altLoc', 'ID2_resName', 'ID2_name']
    df_bi[columns_to_replace_2] = df_links[columns_to_replace_1]
    df_bi[columns_to_replace_1] = df_links[columns_to_replace_2]
    # #combine them
    df_combined = pandas.concat([df_links, df_bi], axis = 0)
    #df_combined.to_csv('bidirectional.csv', index = False)
    return df_combined

# Add self-edges
def add_self_links(file_name, path):
        df_edges = add_bidirectional_links(file_name)
        unique_nodes = df_edges['atom_index1'].unique()
        df_self = pandas.DataFrame(columns= df_edges.columns)
        df_self['atom_index1'] = unique_nodes
        #change index
        df_self['atom_index2'] = unique_nodes        
        # Change attributes 
        # Set attributes to Zero
        df_self['distance'].values[:] = df_self['seq_sep_class'].values[:] = df_self['covalent_bond'].values[:] = df_self['hbond'].values[:] = 0
        # Sum some attributes
        area = []
        boundary = []
        voromqa_energy = []
        for index in unique_nodes:
            area.append(df_edges.loc[df_edges['atom_index1'] == index, 'area'].sum())
            boundary.append(df_edges.loc[df_edges['atom_index1'] == index, 'boundary'].sum()) 
            voromqa_energy.append(df_edges.loc[df_edges['atom_index1'] == index, 'voromqa_energy'].sum())
        df_self['area'] = area
        df_self['boundary'] = boundary
        df_self['voromqa_energy'] = voromqa_energy 
        df_self['is_self'].values[:] = 1
        # Copy atom infos
        for index in unique_nodes:  
            list_1 = ['ID1_chainID', 'ID1_resSeq', 'ID1_iCode', 'ID1_serial', 'ID1_altLoc', 'ID1_resName', 'ID1_name']
            list_2 =['ID2_chainID', 'ID2_resSeq', 'ID2_iCode', 'ID2_serial','ID2_altLoc', 'ID2_resName', 'ID2_name']

            for item in list_1:
                df_self[item] = df_edges.loc[df_edges['atom_index1'] == index, item].iloc[0]
            df_self[list_2] = df_self[list_1]

        # Normalize self_edges:
        columns_to_normalize = ['area', 'boundary', 'voromqa_energy']
        link_means = df_self[columns_to_normalize].mean()
        link_stds = df_self[columns_to_normalize].std()
        # Exclude columns with standard deviation of 0 from normalization
        columns_to_normalize = [col for col in columns_to_normalize if link_stds[col] != 0]
        for index, column in enumerate(columns_to_normalize):
            df_self[column].values[:] = (df_self[column].values[:] - link_means[index]) / link_stds[index]
        
        #combine them
        df_combined = pandas.concat([df_edges, df_self], axis = 0)
        df_combined.to_csv(os.path.join(path,file_name+"_links.csv"), index=False)
         
##################################    
def store_link_input_data():
    files = os.listdir(PATH_ROOT)
    for file in files:
        # Create path
        path = os.path.join(PATH, "prepared_input_data", file)
        if(not os.path.exists(path)):
            os.makedirs(path)
        # Create files 
        add_self_links(file, path)
        
def store_node_input_data():
    files = os.listdir(PATH_ROOT)
    for file in files:
        # Create path
        path = os.path.join(PATH, "prepared_input_data", file)
        if(not os.path.exists(path)):
            os.makedirs(path)  
        normalize_nodes(file)
##################################
#execute
store_node_input_data()
store_link_input_data()
