#Made by @witnull
# TODO:
'''
DONT RUN THIS CODE YET 
NOTHING IS COMFIRMED
'''

import os
import sys

#### SETTINGS #####
SEED = 1



###################

##### Graph Classification #####

# raph Classication for Heterogeous Control Flow Graphs (HCFGs) which detect vulnerabilites at the contract level.
## GAE as node features.
os.system(f"python graph_classifier.py -ld ./logs/graph_classification/cfg/gae/access_control --output_models ./models/graph_classification/cfg/gae/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/ --compressed_graph ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cfg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/graph_labels.json --node_feature gae --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_access_control_cfg_clean_57_0.pkl --seed 1")

# Graph Classication for Heterogeous Call Graphs (HCGs) which detect vulnerabilites at the contract level.
## LINE as node features.
os.system(f"python graph_classifier.py -ld ./logs/graph_classification/cg/line/access_control --output_models ./models/graph_classification/cg/line/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/ --compressed_graph ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/graph_labels.json --node_feature line --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_access_control_cg_clean_57_0.pkl --seed 1")

# Graph Classication for combination of HCFGs and HCGs and which detect vulnerabilites at the contract level.
## node2vec as node features.
os.system(f"python graph_classifier.py -ld ./logs/graph_classification/cg/line/access_control --output_models ./models/graph_classification/cg/line/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/ --compressed_graph ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/cg_compressed_graphs.gpickle --label ./experiments/ge-sc-data/source_code/access_control/clean_57_buggy_curated_0/graph_labels.json --node_feature line --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_access_control_cg_clean_57_0.pkl --seed 1")

##### EO Graph Classification #####

#----------------------------------#

##### Node Classification #####
# Node Classication for Heterogeous Control Flow Graphs (HCFGs) which detect vulnerabilites at the line level.
## GAE as node features for detection access_control bugs.
os.system(f"python node_classifier.py -ld ./logs/node_classification/cfg/gae/access_control --output_models ./models/node_classification/cfg/gae/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/buggy_curated/ --compressed_graph ./experiments/ge-sc-data/source_code/access_control/buggy_curated/cfg_compressed_graphs.gpickle --node_feature gae --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_access_control_cfg_buggy_curated.pkl --testset ./experiments/ge-sc-data/source_code/access_control/curated --seed 1")

# Node Classification for Heterogeous Call Graphs (HCGs) which detect vulnerabilites at the function level.
## The command lines are the same as CFG except the dataset.
## LINE as node features for detection access_control bugs.
os.system(f"python node_classifier.py -ld ./logs/node_classification/cg/line/access_control --output_models ./models/node_classification/cg/line/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/buggy_curated --compressed_graph ./experiments/ge-sc-data/source_code/access_control/buggy_curated/cg_compressed_graphs.gpickle --node_feature line --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_access_control_cg_buggy_curated.pkl --testset ./experiments/ge-sc-data/source_code/access_control/curated --seed 1")

# Node Classication for combination of HCFGs and HCGs and which detect vulnerabilites at the line level.
## node2vec as node features.
os.system(f"python node_classifier.py -ld ./logs/node_classification/cfg_cg/node2vec/access_control --output_models ./models/node_classification/cfg_cg/node2vec/access_control --dataset ./experiments/ge-sc-data/source_code/access_control/buggy_curated --compressed_graph ./experiments/ge-sc-data/source_code/access_control/buggy_curated/cfg_cg_compressed_graphs.gpickle --node_feature node2vec --feature_extractor ./experiments/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_access_control_cfg_cg_buggy_curated.pkl --testset ./experiments/ge-sc-data/source_code/access_control/curated --seed 1")

#We also stack 2 HAN layers for function-level detection. The first HAN layer is based on HCFGs used as feature for the second HAN layer based on HCGs (It will be deprecated in a future version).
os.system(f"python node_classifier.py -ld ./logs/node_classification/call_graph/node2vec_han/access_control --output_models ./models/node_classification/call_graph/node2vec_han/access_control --dataset ./ge-sc-data/node_classification/cg/access_control/buggy_curated --compressed_graph ./ge-sc-data/node_classification/cg/access_control/buggy_curated/compressed_graphs.gpickle --testset ./ge-sc-data/node_classification/cg/curated/access_control --seed 1  --node_feature han --feature_compressed_graph ./data/smartbugs_wild/binary_class_cfg/access_control/buggy_curated/compressed_graphs.gpickle --cfg_feature_extractor ./data/smartbugs_wild/embeddings_buggy_currated_mixed/cfg_mixed/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_access_control_compressed_graphs.pkl --feature_extractor ./models/node_classification/cfg/node2vec/access_control/han_fold_0.pth")


##### EO Node Classification #####


#### Visutalization ####

os.system(f"tensorboard --logdir logs")
########################