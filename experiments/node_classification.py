from math import ceil
import os
import random
import argparse
from os.path import join
from time import time
from torch import nn
import pickle
import json
import networkx as nx
import torch
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from statistics import mean

from sco_models.model_hgt import HGTVulNodeClassifier
#from newMethods.model_test import TEST_MODEL1
#from newMethods.model_test import TEST_MODEL1
from sco_models.utils import get_classification_report
from sco_models.graph_utils import reveert_map_node_embedding, load_hetero_nx_graph

import colorama
from colorama import Fore, Style, Back
# Initialize colorama for Windows compatibility
colorama.just_fix_windows_console()
import sys


# Arguments
parser = argparse.ArgumentParser('MANDO Experiments')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('-e', '--epochs', type=int, default=2,
                    help='Random seed')
parser.add_argument('-rep', '--repeat', type=int, default=2,
                    help='Random seed')
parser.add_argument('-r', '--result', action='store_true')
args = parser.parse_args().__dict__

torch.manual_seed(args['seed'])
random.seed(args['seed'])
ROOT = './experiments'
DATA_ID = 0
REPEAT = args['repeat']
EPOCHS = args['epochs']
LR = 0.001
TASK = "node_classification"
STRUCTURE = 'han'
COMPRESSED_GRAPH = 'cfg_cg'
TRAIN_RATE = 0.7
VAL_RATE = 0.3
ratio = 1


models = ['base_metapath2vec', 'base_gae', 'base_line', 'base_node2vec', 'nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
base_models = ['base_metapath2vec', 'base_gae', 'base_line', 'base_node2vec']
bug_list = ['access_control', 'arithmetic', 'denial_of_service',
            'front_running', 'reentrancy', 'time_manipulation', 
            'unchecked_low_level_calls']
file_counter = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
print(f'Run experiments on {len(bug_list)} kinds of bug for {len(models)} kinds of model.')
print(f'Repeat {REPEAT} times and {EPOCHS} epochs for each experiment.')


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask.byte()


def get_node_ids(graph, source_files):
    file_ids = []
    for node_ids, node_data in graph.nodes(data=True):
        filename = node_data['source_file']
        if filename in source_files:
            file_ids.append(node_ids)
    return file_ids


def get_node_label(nx_graph):
    nx_g = nx_graph
    node_labels = []
    label_ids = {'valid': 0}
    labeled_node_ids = {'buggy': [], 'valid': []}
    for node_id, node_data in nx_g.nodes(data=True):
        node_type = node_data['node_type']
        node_label = node_data['node_info_vulnerabilities']
        target = 0
        if node_label is None:
            target = 0
            labeled_node_ids['valid'].append(node_id)
        else:
            bug_type = node_label[0]['category']
            if bug_type not in label_ids:
                label_ids[bug_type] = len(label_ids)
            target = 1
            labeled_node_ids['buggy'].append(node_id)
        node_labels.append(target)
    return node_labels, labeled_node_ids, label_ids


def base_metapath2vec(compressed_graph, source_path, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=None, node_feature='metapath2vec', device=device)

    # Check graph structure
    # print(f"Graph nodes: {model.symmetrical_global_graph.number_of_nodes()}")
    # print(f"Graph edges: {model.symmetrical_global_graph.number_of_edges()}")
    # print(f"Node types: {model.symmetrical_global_graph.ntypes}")
    # print(f"Edge types: {model.symmetrical_global_graph.etypes}")
    # print(f"Available node features: {model.symmetrical_global_graph.ndata.keys()}")

    features = model.symmetrical_global_graph.ndata['feat']

    # print(f'Features:{features}\n\n')

    nx_graph = load_hetero_nx_graph(compressed_graph)
    embedding = reveert_map_node_embedding(nx_graph, features)
    embedding = torch.tensor(embedding, device=device) 
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]

    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    t0 = time()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    t1 = time()

    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_metapath2vec/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    save_path = os.path.join(output_models, f'model.pth')
    torch.save(model.state_dict(), save_path)


    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_gae(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    # print(embedded)
    # print(len(nx_graph.nodes))
    # print(embedding.shape)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    t0 = time()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    t1 = time()

    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_gae/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    save_path = os.path.join(output_models, f'model.pth')
    torch.save(model.state_dict(), save_path)



    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_line(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
   
 
    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()
    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    t0 = time()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    t1 = time()

    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_line/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    save_path = os.path.join(output_models, f'model.pth')
    torch.save(model.state_dict(), save_path)

    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def base_node2vec(nx_graph, embedded, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)


    with open(embedded, 'rb') as f:
        embedding = pickle.load(f, encoding="utf8")
        embedding = torch.tensor(embedding, device=device)
    assert len(nx_graph.nodes) == embedding.shape[0]
    number_of_nodes = embedding.shape[0]
    train, test = train_test_split(range(number_of_nodes), test_size=VAL_RATE)
    train_mask = get_binary_mask(number_of_nodes, train)
    test_mask = get_binary_mask(number_of_nodes, test)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()
    classifier = torch.nn.Linear(128, 2)
    classifier.to(device)
    classifier.train()

    targets, _, _ = get_node_label(nx_graph)
    targets = torch.tensor(targets, device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    t0 = time()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = classifier(embedding)
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask])
        train_loss.backward()
        optimizer.step()
    t1 = time()

    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/base_node2vec/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    save_path = os.path.join(output_models, f'model.pth')
    torch.save(model.state_dict(), save_path)



    test_results = get_classification_report(targets[test_mask], logits[test_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def train(model, train_mask, targets, device):
    total_steps = EPOCHS
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, total_steps=total_steps)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model()
        logits = logits.to(device)
        train_loss = loss_fcn(logits[train_mask], targets[train_mask]) 
        train_loss.backward()
        optimizer.step()
        scheduler.step()
    return model


def nodetype(compressed_graph, source_code, dataset, bugtype, device, test_only=False):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/buggy_curated/'
    test_logs = f'./newMethods/logs/'# f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/buggy_curated/test_logs/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    if not os.path.exists(test_logs):
        os.makedirs(test_logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    
    feature_extractor = None
    node_feature = 'nodetype'
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    train_mask, val_mask, verify_mask = dataset
    targets = torch.tensor(model.node_labels, device=device)
    model.to(device)
    model.reset_parameters()
    t0 = time()
    save_path = os.path.join(output_models, f'han.pth')

    if test_only == False:
        logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/buggy_curated/' 
        model = train(model, train_mask, targets, device)
        torch.save(model.state_dict(), save_path)
    else:
        print("Testing........")
        logs = f'./newMethods/logs/'#f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/nodetype/{bugtype}/buggy_curated/test_logs/' #testlogs
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.to(device)

    t1 = time()
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_results={}
        if test_only == False:
            #print(logits)
            #print(logits[val_mask])

            test_results = get_classification_report(targets[val_mask], logits[val_mask], output_dict=True)
        else:
            print(get_classification_report(targets[val_mask], logits[val_mask], output_dict=True))
            print(targets[0])
            print(logits[0])
            print('Verify mask:', verify_mask)
            print("Val mask:",logits[val_mask].shape)
            print("Label shape:",targets[val_mask].shape)
            logits = nn.functional.softmax(logits[val_mask], dim=1)
            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            #test_results['predictions'] = prediction.tolist()
            print('Verify mask:', len(prediction.tolist()))

            #print("Real val mask:", val_mask.shape)
            #test_results = get_classification_report(targets[verify_mask], logits[verify_mask], output_dict=True)

    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def metapath2vec(compressed_graph, source_code, dataset, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/metapath2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    feature_extractor = None
    node_feature = 'metapath2vec'
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    train_mask, val_mask = dataset
    targets = torch.tensor(model.node_labels, device=device)
    model.to(device)
    model.reset_parameters()
    t0 = time()
    model = train(model, train_mask, targets, device)
    save_path = os.path.join(output_models, f'han.pth')
    torch.save(model.state_dict(), save_path)
    t1 = time()
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_results = get_classification_report(targets[val_mask], logits[val_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def gae(compressed_graph, source_code, dataset, feature_extractor, bugtype, device):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/gae/{bugtype}/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    if not os.path.exists(output_models):
        os.makedirs(output_models)
    # feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'gae'
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    train_mask, val_mask = dataset
    targets = torch.tensor(model.node_labels, device=device)
    model.to(device)
    model.reset_parameters()
    t0 = time()
    model = train(model, train_mask, targets, device)
    save_path = os.path.join(output_models, f'han.pth')
    torch.save(model.state_dict(), save_path)
    t1 = time()
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_results = get_classification_report(targets[val_mask], logits[val_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def line(compressed_graph, source_code, dataset, feature_extractor, bugtype, device,test_only=False):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/buggy_curated/'
    test_logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/buggy_curated/test_logs/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    if not os.path.exists(test_logs):
        os.makedirs(test_logs)
    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)

    # feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'line'
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    train_mask, val_mask = dataset
    targets = torch.tensor(model.node_labels, device=device)
    model.to(device)
    model.reset_parameters()
    
    save_path = os.path.join(output_models, f'han.pth')
    if test_only == False:
        logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/buggy_curated/' 
        t0 = time()
        model = train(model, train_mask, targets, device)
        torch.save(model.state_dict(), save_path)
        t1 = time()
    else:
        logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/line/{bugtype}/buggy_curated/test_logs/' #testlogs
        t0 = time()
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.to(device)
        t1 = time()

    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_results = get_classification_report(targets[val_mask], logits[val_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def node2vec(compressed_graph, source_code, dataset, feature_extractor, bugtype, device, test_only = False):
    logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/buggy_curated/'
    test_logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(logs):
        os.makedirs(logs)
    if not os.path.exists(test_logs):
        os.makedirs(test_logs)

    output_models = f'{ROOT}/models/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/buggy_curated/'
    if not os.path.exists(output_models):
        os.makedirs(output_models)

    # feature_extractor = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_clean_{file_counter[bugtype]}_{DATA_ID}.pkl'
    node_feature = 'node2vec'
    model = HGTVulNodeClassifier(compressed_graph, feature_extractor=feature_extractor, 
                                 node_feature=node_feature, device=device)
    train_mask, val_mask = dataset
    targets = torch.tensor(model.node_labels, device=device)
    model.to(device)
    model.reset_parameters()

    save_path = os.path.join(output_models, f'han.pth')
    if test_only == False:
        logs = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/buggy_curated/' 
        t0 = time()
        model = train(model, train_mask, targets, device)
        torch.save(model.state_dict(), save_path)
        t1 = time()
    else:
        logs = f'{ROOT}/testlogs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/node2vec/{bugtype}/buggy_curated/' #testlogs
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.to(device)
    #TEST 
    model.eval()
    with torch.no_grad():
        logits = model()
        logits = logits.to(device)
        test_results = get_classification_report(targets[val_mask], logits[val_mask], output_dict=True)
    t2 = time()
    test_results['train_time'] = str(t1 - t0)
    test_results['test_time'] = str(t2 - t1)
    if os.path.isfile(join(logs, 'test_report.json')):
        with open(join(logs, 'test_report.json'), 'r') as f:
            report = json.load(f)
        report.append(test_results)
    else:
        report = [test_results]
    with open(join(logs, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


def full_test_execution(compressed_graph, source_code, dataset, line_embedded, gae_embedded, node2vec_embedded, bugtype, device):
   nodetype(compressed_graph, source_code, dataset, bugtype, device, True)



def print_info():
    '''
    Print information about the script.
    Color rule:
    Success: Green
    Warning: Yellow , or RED background
    Error: Red
    Info: Cyan
    Action: Blue
    '''
    print("\n\n ###################### INFO ##################### \n\n")
    print(f"Usage: python -m experiments.node_classification --epoch X --repeat Y")
    print(f"{Back.CYAN}{Fore.BLUE}  This script is node classification. PHASE 2 Fine-grained detection. Table III in paper \n{Style.RESET_ALL}")
    print(f"###### BASELINES: \n")
    print(f"{Fore.YELLOW} Remember to run process_graphs/byte_code_control_flow_graph_generator.py first .... ")

    print(f"{Fore.CYAN} [1] Run baselines base_metapath2vec . OK - metapath2vec {Style.RESET_ALL}")
    print(f"{Fore.CYAN} [2] Run baselines base_gae . OK - fast{Style.RESET_ALL}") 
    print(f"{Fore.CYAN} [3] Run baselines base_line . OK - fast{Style.RESET_ALL}")
    print(f"{Fore.CYAN} [4] Run baselines base_node2vec . OK - fast{Style.RESET_ALL}\n")

    print(f"###### MANDO-HGT: \n")
    print(f"{Fore.CYAN} [5] Run MandoHGT nodetype (Default). OK - slow{Style.RESET_ALL}")
    print(f"{Fore.CYAN} [6] Run MandoHGT metapath2vec. OK - slow {Style.RESET_ALL}")
   # print(f"{Fore.CYAN} [7] Run MandoHGT gae. OK - slow{Style.RESET_ALL}")
    print(f"{Fore.CYAN} [8] Run MandoHGT line. OK - slow{Style.RESET_ALL}")
    print(f"{Fore.CYAN} [9] Run MandoHGT node2vec.  OK - slow{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW} Modify the script to change paths and bug types.{Style.RESET_ALL}")

    print(f"####### FULL TEST: \n")
    print(f"{Fore.CYAN} [ft] Run test{Style.RESET_ALL}")
    # Function to wait for user input before proceeding

    #print(f"\n\n {Back.RED}!!! Warning: After run remember to copy the log/results cuz it might get deleted when u run the train once again.{Style.RESET_ALL}\n\n")
    
    option_list = ['1', '2','3','4','5','6','7','8','9','ft']
    option = '4'
    option = input(f"{Back.BLUE}Please input your option ({', '.join(option_list)}):{Style.RESET_ALL}")
    if option not in option_list:       
        print(f"{Fore.RED}Invalid option. Please input {', '.join(option_list)}.{Style.RESET_ALL}")
        sys.exit(1)
    return option

def main(device):
    option = print_info()

    for bugtype in bug_list:
        print('Bugtype {}'.format(bugtype))
        for i in range(REPEAT):
            print(f'Processing bugtype {bugtype} {i}-th/{REPEAT} reppeat ')
            compressed_graph = f'{ROOT}/ge-sc-data/source_code/{bugtype}/buggy_curated/{COMPRESSED_GRAPH}_compressed_graphs.gpickle'
            nx_graph = nx.read_gpickle(compressed_graph)
            number_of_nodes = len(nx_graph)
            source_path = f'{ROOT}/ge-sc-data/source_code/{bugtype}/buggy_curated'
            testset = f'{ROOT}/ge-sc-data/source_code/{bugtype}/curated'
            
          

            #############################

            total_train_files = [f for f in os.listdir(source_path) if f.endswith('.sol')]
            total_test_files = [f for f in os.listdir(testset) if f.endswith('.sol')]

            total_train_files = list(set(total_train_files).difference(set(total_test_files)))

            rand_train_ids = torch.randperm(len(total_train_files)).tolist()
            rand_test_ids = torch.randperm(len(total_test_files)).tolist()

            train_size_0 = int(TRAIN_RATE * len(total_train_files))
            train_size_1 = int(TRAIN_RATE * len(total_test_files))
            train_files = [total_train_files[i] for i in rand_train_ids[:train_size_0]] + \
                        [total_test_files[i] for i in rand_test_ids[:train_size_1]]

            val_size_0 = len(total_train_files) - train_size_0
            val_size_1 = len(total_test_files) - train_size_1
            val_files = [total_train_files[i] for i in rand_train_ids[train_size_0:train_size_0 + val_size_0]] + \
                        [total_test_files[i] for i in rand_test_ids[train_size_1:train_size_1 + val_size_1]]

            assert len(train_files) + len(val_files) == len(total_train_files) + len(total_test_files)

            train_ids = get_node_ids(nx_graph, train_files)
            val_ids = get_node_ids(nx_graph, val_files)
            train_mask = get_binary_mask(number_of_nodes, train_ids)
            val_mask = get_binary_mask(number_of_nodes, val_ids)

            #Verify set

            verify_set = f'./newMethods/sampleDataset'
            verify_compressed_graph = f'./newMethods/sampleDataset/{COMPRESSED_GRAPH}_compressed_graphs.gpickle'
            verify_nx_graph = nx.read_gpickle(verify_compressed_graph)
            verify_number_of_nodes = len(verify_nx_graph)
            total_verify_files = [f for f in os.listdir(verify_set) if f.endswith('.sol')]
            verify_ids = get_node_ids(verify_nx_graph, total_verify_files)
            verify_mask = get_binary_mask(verify_number_of_nodes, verify_ids)

            #####################
            if hasattr(torch, 'BoolTensor'):
                train_mask = train_mask.bool()
                val_mask = val_mask.bool()
                if verify_mask is not None:
                    verify_mask = verify_mask.bool()
            dataset = (train_mask, val_mask, verify_mask)


            gae_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_gae_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated.pkl'
            line_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_line_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated.pkl'
            node2vec_embedded = f'{ROOT}/ge-sc-data/source_code/gesc_matrices_node_embedding/matrix_node2vec_dim128_of_core_graph_of_{bugtype}_{COMPRESSED_GRAPH}_buggy_curated.pkl'
            # Base line
            if (option == '1'):
                base_metapath2vec(compressed_graph, source_path, bugtype, device)
            elif (option == '2'):
                base_gae(nx_graph, gae_embedded, bugtype, device)
            elif (option == '3'):
                base_line(nx_graph, line_embedded, bugtype, device)
            elif (option == '4'):
                base_node2vec(nx_graph, gae_embedded, bugtype, device)

            # Our models
            elif (option == '5'):
                nodetype(compressed_graph, source_path, dataset, bugtype, device)
            elif (option == '6'):
                metapath2vec(compressed_graph, source_path, dataset, bugtype, device)
            elif (option == '7'):
                gae(compressed_graph, source_path, dataset, gae_embedded, bugtype, device)
            elif (option == '8'):
                line(compressed_graph, source_path,  dataset, line_embedded, bugtype, device,True)
            elif (option == '9'):
                node2vec(compressed_graph, source_path, dataset, line_embedded, bugtype, device)
            elif (option == '10'):
                print("not implemented")
               # test_model1(compressed_graph, source_path, dataset, line_embedded, bugtype, device)
            elif (option == 'ft'):
                full_test_execution(verify_compressed_graph, source_path, dataset, line_embedded, gae_embedded, node2vec_embedded , bugtype, device)
            else:
                print("Invalid option!")
                sys.exit(1)


def get_avg_results(report_path, top_rate=0.5):
    with open(report_path, 'r') as f:
        results = json.load(f)
    buggy_f1 = []
    macro_f1 = []
    for i in range(len(results)):
        buggy_f1.append(results[i]['1']['f1-score'])
        macro_f1.append(results[i]['macro avg']['f1-score'])
    return round(mean(sorted(buggy_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2), round(mean(sorted(macro_f1, reverse=True)[:int(top_rate*len(results))]) * 100, 2)


def get_max_results(report_path):
    with open(report_path, 'r') as f:
        results = json.load(f)
    buggy_f1 = []
    macro_f1 = []
    for i in range(len(results)):
        buggy_f1.append(results[i]['1']['f1-score'])
        macro_f1.append(results[i]['macro avg']['f1-score'])
    return round(max(buggy_f1) * 100, 2), round(max(macro_f1) * 100, 2)


def get_results():
    buggy_f1_report = {}
    macro_f1_report = {}
    for bugtype in bug_list:
        for model in models:
            report_path = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/buggy_curated/test_report.json'
            if not os.path.exists(report_path):
                print(f'REPORT {report_path} not found!')
                continue
            try:
                buggy_f1, macro_f1 = get_avg_results(report_path, top_rate=0.5)
              #  print(f'{Fore.GREEN}REPORT {report_path} found!{Style.RESET_ALL}')
            except:
                buggy_f1, macro_f1 = 0.0, 0.0
            # buggy_f1, macro_f1 = get_max_results(report_path)
            if model not in buggy_f1_report:
                buggy_f1_report[model] = [buggy_f1]
                macro_f1_report[model] = [macro_f1]
            else:
                buggy_f1_report[model].append(buggy_f1)
                macro_f1_report[model].append(macro_f1)
    data = []
    for model in models:
        # print(' ', end=' ')
        # print(' \t'.join(['%.2f'%n for n in buggy_f1_report[model]]), end=r'')
        # print()
        # print(' ', end=' ')
        # print(' \t'.join(['%.2f'%n for n in macro_f1_report[model]]), end=r'')
        # print()
        buggy_f1_row = []
        macro_f1_row = []
        if model not in buggy_f1_report:
            continue
        for i in range(len(buggy_f1_report[model])):
            buggy_f1 = buggy_f1_report[model][i]
            macro_f1 = macro_f1_report[model][i]
            buggy_f1_row.append('%.2f'%buggy_f1 + '%' if isinstance(buggy_f1, float) else buggy_f1)
            macro_f1_row.append('%.2f'%macro_f1 + '%' if isinstance(macro_f1, float) else macro_f1)
        data.append([model, 'Buggy-F1'] + buggy_f1_row)
        data.append([model, 'Macro-F1'] + macro_f1_row)
    print(f"{Fore.GREEN}")
    print(tabulate(data, headers=bug_list, tablefmt='orgtbl'))
    print(f"{Style.RESET_ALL}")



def get_test_results():
    buggy_f1_report = {}
    macro_f1_report = {}
    models = ['nodetype', 'metapath2vec', 'gae', 'line', 'node2vec']
    for bugtype in bug_list:
        for model in models:
            report_path = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/buggy_curated/test_logs/test_report.json'
            if not os.path.exists(report_path):
                #print(f'REPORT {report_path} not found!')
                continue
            try:
                buggy_f1, macro_f1 = get_avg_results(report_path, top_rate=0.5)
              #  print(f'{Fore.GREEN}REPORT {report_path} found!{Style.RESET_ALL}')
            except:
                buggy_f1, macro_f1 = 0.0, 0.0
            # buggy_f1, macro_f1 = get_max_results(report_path)
            if model not in buggy_f1_report:
                buggy_f1_report[model] = [buggy_f1]
                macro_f1_report[model] = [macro_f1]
            else:
                buggy_f1_report[model].append(buggy_f1)
                macro_f1_report[model].append(macro_f1)
    data = []
    for model in models:
        # print(' ', end=' ')
        # print(' \t'.join(['%.2f'%n for n in buggy_f1_report[model]]), end=r'')
        # print()
        # print(' ', end=' ')
        # print(' \t'.join(['%.2f'%n for n in macro_f1_report[model]]), end=r'')
        # print()
        buggy_f1_row = []
        macro_f1_row = []
        if model not in buggy_f1_report:
            continue
        for i in range(len(buggy_f1_report[model])):
            buggy_f1 = buggy_f1_report[model][i]
            macro_f1 = macro_f1_report[model][i]
            buggy_f1_row.append('%.2f'%buggy_f1 + '%' if isinstance(buggy_f1, float) else buggy_f1)
            macro_f1_row.append('%.2f'%macro_f1 + '%' if isinstance(macro_f1, float) else macro_f1)
        data.append([model, 'Buggy-F1'] + buggy_f1_row)
        data.append([model, 'Macro-F1'] + macro_f1_row)
    print(f"{Fore.GREEN}")
    print(tabulate(data, headers=bug_list, tablefmt='orgtbl'))
    print(f"{Style.RESET_ALL}")


def get_exp_time(report_path):
    with open(report_path, 'r') as f:
        results = json.load(f)
    train_time = []
    test_time = []
    for i in range(len(results)):
        train_time.append(float(results[i]['train_time']))
        test_time.append(float(results[i]['test_time']))
    return round(mean(train_time), 2), round(mean(test_time), 2)

def get_runtime_result():
    train_time_report = {}
    test_time_report = {}
    for bugtype in bug_list:
        for model in models:
            report_path = f'{ROOT}/logs/{TASK}/{STRUCTURE}/{COMPRESSED_GRAPH}/{model}/{bugtype}/buggy_curated/test_report.json'
            train_time, test_time = get_exp_time(report_path)
            # train_time, test_time = get_max_results(report_path)
            if model not in train_time_report:
                train_time_report[model] = [train_time]
                test_time_report[model] = [test_time]
            else:
                train_time_report[model].append(train_time)
                test_time_report[model].append(test_time)
    avg_train_time = []
    avg_test_time = []
    for i in range(len(bug_list)):
        bug_train_list = []
        bug_test_list = []
        for model in models[-5:]:
            bug_train_list.append(train_time_report[model][i])
            bug_test_list.append(test_time_report[model][i])
        avg_train_time.append(mean(bug_train_list))
        avg_test_time.append(mean(bug_test_list))
    print(avg_train_time)
    for i in range(len(bug_list)):
        print(f'{ceil(avg_train_time[i])}/{ceil(avg_test_time[i])}', end=' ')
        print('&', end = ' ')


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args['result']:
        print('\n TRAIN RESULTS \n')
        get_results()
        print('\n\n\n TEST RESULT \n\n\n')
        get_test_results()
    else:
        print(device)
        main(device)
