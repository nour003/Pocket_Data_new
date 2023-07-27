import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

ROOT_PATH = sys.argv[1]
#################################
# Save string to file 
def save_to_file(path, list):
    with open(path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(list))
        myfile.write('\n')

#################################
# Read strings in file
def read_strings(path):
    with open(path) as file:
        strings=[line.rstrip() for line in file]
    return strings

# Read floats
def read_floats(path):
    with open(path) as file:
        floats=[float(line.rstrip()) for line in file]
    return floats

#################################
# split data into train/validation 
def split_data():
    # get all files in folder 
    path = os.path.join(ROOT_PATH, 'separated_graphs')
    files = os.listdir(path)
    N = len(files)
    # split randomly protein class names into train/validation lists 
    x_train = random.sample(files, k=round(N * 0.8))
    for x in x_train:
        files.remove(x)
    x_val = random.sample(files, k=round(N * 0.2))
    # Write each list to output location
    prefixes = ["training_data_files_prefixes", "validation_data_files_prefixes"]
    data_list=[x_train, x_val]
    for f in range(len(prefixes)):
        save_to_file(os.path.join(ROOT_PATH, prefixes[f]), data_list[f])

#################################
#Group all training data:
def group_data(dataset_name):
    if(dataset_name == 'train_data'):
        raw_prefixes= read_strings( ROOT_PATH+"/training_data_files_prefixes")
    elif(dataset_name == 'validation_data'):
        raw_prefixes= read_strings(ROOT_PATH+"/validation_data_files_prefixes")
    raw_paths = [os.path.join(ROOT_PATH,"separated_graphs", pref,pref) for pref in raw_prefixes]
    file_suffix = ["nodes", "links"]
    for suffix in file_suffix:
        joined_list = [(path + "_" + suffix + ".csv") for path in raw_paths]
        dfs= []
        for file in joined_list:
            data = pd.read_csv(file)
            dfs.append(data)
        merged_df = pd.concat(dfs, axis =0, ignore_index= True)
        file_name = "merged_"+ suffix +"_files.csv"
        path = os.path.join(ROOT_PATH, dataset_name, file_name)
        merged_df.to_csv(path, index=False)

#################################
# Get mean of training data
def get_means(dataset_name, file_suffix):
    file_name = "merged_"+file_suffix +"_files.csv"
    path = os.path.join(ROOT_PATH, dataset_name, file_name)
    if (not os.path.exists(path)):
        group_data(dataset_name)
    df = pd.read_csv(path)
    if (file_suffix == "nodes"):
        columns=['radius', 'voromqa_sas_potential','residue_mean_sas_potential', 'residue_sum_sas_potential','residue_size','sas_area', 'voromqa_sas_energy','voromqa_depth','voromqa_score_a', 'voromqa_score_r', 'volume', \
                    'volume_vdw', 'ufsr_a1', 'ufsr_a2', 'ufsr_a3', 'ufsr_b1','ufsr_b2', 'ufsr_b3', 'ufsr_c1', 'ufsr_c2', 'ufsr_c3', 'ev28', 'ev56','ground_truth']
    elif(file_suffix == "links"):
        columns=['area', 'boundary', 'distance', 'voromqa_energy', 'seq_sep_class', 'covalent_bond', 'hbond']
    else :
        raise Exception("no such file_suffix exist!")
    return df[columns].mean(axis = 0)

# Get standard deviations of training data
def get_stds(dataset_name, file_suffix):
    file_name = "merged_"+file_suffix +"_files.csv"
    path = os.path.join(ROOT_PATH, dataset_name, file_name)    
    if (not os.path.exists(path)):
        group_data(dataset_name) 
    df = pd.read_csv(path)
    if (file_suffix == "nodes"):
        columns=['radius', 'voromqa_sas_potential','residue_mean_sas_potential', 'residue_sum_sas_potential','residue_size','sas_area', 'voromqa_sas_energy','voromqa_depth','voromqa_score_a', 'voromqa_score_r', 'volume', \
                    'volume_vdw', 'ufsr_a1', 'ufsr_a2', 'ufsr_a3', 'ufsr_b1','ufsr_b2', 'ufsr_b3', 'ufsr_c1', 'ufsr_c2', 'ufsr_c3', 'ev28', 'ev56','ground_truth']
    elif(file_suffix == "links"):
        columns=['area', 'boundary', 'distance', 'voromqa_energy', 'seq_sep_class', 'covalent_bond', 'hbond']
    else :
        raise Exception("no such file_suffix exist!")
    return df[columns].std(axis = 0)

############################################
# Write means of training data into a file
def write_means(dataset_name):
    # Links means
    means = get_means(dataset_name,"nodes")
    # Write means
    path = os.path.join(ROOT_PATH, dataset_name, "means")
    if(not os.path.exists(path)):
            os.makedirs(path)
     
    file_path = os.path.join(path, 'nodes_mean.txt')
    with open(file_path, 'w') as f:
        f.writelines([str(mean)+'\n' for mean in means ])
    # Node means
    means = get_means(dataset_name,"links")
    file_path = os.path.join(path, "links_mean.txt")
    with open(file_path, 'w') as f:
        f.writelines([str(mean)+'\n' for mean in means ])

# Write stds of training data into a file
def write_stds(dataset_name):
    path = os.path.join(ROOT_PATH,dataset_name,"stds")
    if(not os.path.exists(path)):
            os.makedirs(path)
    # Link std
    stds = get_stds(dataset_name,"nodes")
    # Write stds
    
    file_path = os.path.join(path, 'nodes_std.txt')
    with open(file_path, 'w') as f:
        f.writelines([str(std)+'\n' for std in stds ])

    # Node std
    stds = get_stds(dataset_name,"links")
    file_path = os.path.join(path, "links_std.txt")
    with open(file_path, 'w') as f:
        f.writelines([str(std)+'\n' for std in stds ])

############################################
# Distribution of the ground truth
def ground_truth_dist(dataset_name):
    group_data(dataset_name)
    path = os.path.join(ROOT_PATH, "train_data", "merged_nodes_files.csv")
    df_node = pd.read_csv(path)
    x = df_node['ground_truth']
    ax = sns.distplot(x)
    plt.axvline(x=0.1)
    plt.savefig('Ground_truth_distribution.png')

#split_data()
write_means('train_data')
write_stds('train_data')
