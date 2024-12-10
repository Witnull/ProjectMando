import os
import json

from os.path import join
from shutil import copy
from copy import deepcopy
from re import L
from typing import Pattern
from tqdm import tqdm
import re

import networkx as nx
from slither.slither import Slither
from slither.core.cfg.node import NodeType

import subprocess
import sys
import logging

import colorama
from colorama import Fore, Style, Back
# Initialize colorama for Windows compatibility
colorama.just_fix_windows_console()

#####################################################
# Solc version / download solc
#####################################################

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def get_solc_version(file_path):
    pattern = re.compile(r"pragma solidity\s*(?:\^|>=|<=)?\s*(\d+\.\d+\.\d+)")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'pragma solidity' in line:
                match = pattern.search(line)
                if match:
                    return match.group(1)
    return '0.4.25'  # Default version if not found

def install_solc_version(version, logger):
    try:
        logger.info(f"Installing solc version {version}")
        subprocess.run(['solc-select', 'install', version], check=True)
        logger.info(f"Successfully installed solc version {version}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install solc version {version}: {e}")
        sys.exit(1)

def set_solc_version(version, logger):
    try:
        logger.info(f"Setting solc version to {version}")
        subprocess.run(['solc-select', 'use', version], check=True)
    except subprocess.CalledProcessError:
        logger.warning(f"solc version {version} not found. Attempting to install...")
        install_solc_version(version, logger)
        set_solc_version(version, logger)  # Retry setting the version after installation

######################################################
######################################################


def get_node_info(node, list_vulnerabilities_info_in_sc):
    node_label = "Node Type: {}\n".format(str(node.type))
    node_type = str(node.type)
    if node.expression:
        node_label += "\nEXPRESSION:\n{}\n".format(node.expression)
        node_expression = str(node.expression)
    else:
        node_expression = None
    if node.irs:
        node_label += "\nIRs:\n" + "\n".join([str(ir) for ir in node.irs])
        node_irs = "\n".join([str(ir) for ir in node.irs])
    else:
        node_irs = None

    # Get the source mapping lines safely
    if hasattr(node.source_mapping, 'lines'):
        node_source_code_lines = node.source_mapping.lines
    else:
        node_source_code_lines = []

    node_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_source_code_lines, list_vulnerabilities_info_in_sc)
    
    return node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines

def get_vulnerabilities(file_name_sc, vulnerabilities):
    list_vulnerability_in_sc = None
    if vulnerabilities is not None:
        for vul_item in vulnerabilities:
            if file_name_sc == vul_item['name']:
                list_vulnerability_in_sc = vul_item['vulnerabilities']
            
    return list_vulnerability_in_sc

def get_vulnerabilities_of_node_by_source_code_line(source_code_lines, list_vul_info_sc):
    if list_vul_info_sc is not None:
        list_vulnerability = []
        for vul_info_sc in list_vul_info_sc:
            vulnerabilities_lines = vul_info_sc['lines']
            # for source_code_line in source_code_lines:
            #     for vulnerabilities_line in vulnerabilities_lines:
            #         if source_code_line == vulnerabilities_line:
            #             list_vulnerability.append(vul_info_sc)
            interset_lines = set(vulnerabilities_lines).intersection(set(source_code_lines))
            if len(interset_lines) > 0:
                list_vulnerability.append(vul_info_sc)

    else:
        list_vulnerability = None
    
    if list_vulnerability is None or len(list_vulnerability) == 0:
        node_info_vulnerabilities = None
    else:
        node_info_vulnerabilities = list_vulnerability

    return node_info_vulnerabilities

def compress_full_smart_contracts(smart_contracts, input_graph, output, vulnerabilities=None):
    logger = setup_logging()
    full_graph = None
    if input_graph is not None:
        full_graph = nx.read_gpickle(input_graph)
    count = 0
    for sc in tqdm(smart_contracts):
        sc_version = get_solc_version(sc)
        # print(f'{sc} - {sc_version}')
        # solc_compiler = f'.solc-select/artifacts/solc-{sc_version}'
        # if not os.path.exists(solc_compiler):
        #     solc_compiler = f'.solc-select/artifacts/solc-0.4.25'
        set_solc_version(sc_version, logger)
        file_name_sc = sc.split('/')[-1:][0]
        bug_type = sc.split('/')[-2]
        try:
            slither = Slither(sc)
            count += 1
        except Exception as e:
            print('exception ', e)
            continue

        list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

        print(file_name_sc, list_vul_info_sc)

        merge_contract_graph = None
        for contract in slither.contracts:
            merged_graph = None
            for idx, function in enumerate(contract.functions + contract.modifiers):  

                nx_g = nx.MultiDiGraph()
                for nidx, node in enumerate(function.nodes):             
                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                    
                    nx_g.add_node(node.node_id, label=node_label,
                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                  node_source_code_lines=node_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                    
                    if node.type in [NodeType.IF, NodeType.IFLOOP]:
                        true_node = node.son_true
                        if true_node:
                            if true_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                                nx_g.add_node(true_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                        
                        
                        false_node = node.son_false
                        if false_node:
                            if false_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                                nx_g.add_node(false_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                            
                    else:
                        for son_node in node.sons:
                            if son_node:
                                if son_node.node_id not in nx_g.nodes():
                                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                    nx_g.add_node(son_node.node_id, label=node_label,
                                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                                  node_source_code_lines=node_source_code_lines,
                                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                                nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

                nx_graph = nx_g
                # add FUNCTION_NAME node
                node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
                if hasattr(function.source_mapping, 'lines'):
                    node_function_source_code_lines = function.source_mapping.lines
                else:
                    node_function_source_code_lines = []
                node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
                nx_graph.add_node(node_function_name, label=node_function_name,
                                  node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                  node_info_vulnerabilities=node_function_info_vulnerabilities,
                                  node_source_code_lines=node_function_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                
                if 0 in nx_graph.nodes():
                    nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')

                nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

                if merged_graph is None:
                    merged_graph = deepcopy(nx_graph)
                else:
                    merged_graph = nx.disjoint_union(merged_graph, nx_graph)

            if merge_contract_graph is None:
                merge_contract_graph = deepcopy(merged_graph)
            elif merged_graph is not None:
                merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
        
        if full_graph is None:
            full_graph = deepcopy(merge_contract_graph)
        elif merge_contract_graph is not None:
            full_graph = nx.disjoint_union(full_graph, merge_contract_graph)

    # for node, node_data in full_graph.nodes(data=True):
    #     if node_data['node_info_vulnerabilities'] is not None:
    #         print('Node has vulnerabilities:', node, node_data)
    print(f'{count}/{len(smart_contracts)}')
    # nx.nx_agraph.write_dot(full_graph, output.replace('.gpickle', '.dot'))
    nx.write_gpickle(full_graph, output)

def merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files):
    result = list()
    for f1 in list_vulnerabilities_json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    return result

def check_extract_graph(source_path):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    try:
        slither = Slither(source_path, solc=solc_compiler)
        return 1
    except Exception as e:
        return 0


def extract_graph(source_path, output, vulnerabilities=None):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    file_name_sc = source_path.split('/')[-1]
    try:
        slither = Slither(source_path, solc=solc_compiler)
    except Exception as e:
        print('exception ', e)
        return 0

    list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

    merge_contract_graph = None
    for contract in slither.contracts:
        merged_graph = None
        for idx, function in enumerate(contract.functions + contract.modifiers):  

            nx_g = nx.MultiDiGraph()
            for nidx, node in enumerate(function.nodes):             
                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                
                nx_g.add_node(node.node_id, label=node_label,
                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                node_info_vulnerabilities=node_info_vulnerabilities,
                                node_source_code_lines=node_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                
                if node.type in [NodeType.IF, NodeType.IFLOOP]:
                    true_node = node.son_true
                    if true_node:
                        if true_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                            nx_g.add_node(true_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                    
                    
                    false_node = node.son_false
                    if false_node:
                        if false_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                            nx_g.add_node(false_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                        
                else:
                    for son_node in node.sons:
                        if son_node:
                            if son_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                nx_g.add_node(son_node.node_id, label=node_label,
                                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                node_info_vulnerabilities=node_info_vulnerabilities,
                                                node_source_code_lines=node_source_code_lines,
                                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

            nx_graph = nx_g
            # add FUNCTION_NAME node
            node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
            if hasattr(function.source_mapping, 'lines'):
                node_function_source_code_lines = function.source_mapping.lines
            else:
                node_function_source_code_lines = []
            node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
            nx_graph.add_node(node_function_name, label=node_function_name,
                                node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                node_info_vulnerabilities=node_function_info_vulnerabilities,
                                node_source_code_lines=node_function_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
            
            if 0 in nx_graph.nodes():
                nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')

            nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

            if merged_graph is None:
                merged_graph = deepcopy(nx_graph)
            else:
                merged_graph = nx.disjoint_union(merged_graph, nx_graph)

        if merge_contract_graph is None:
            merge_contract_graph = deepcopy(merged_graph)
        elif merged_graph is not None:
            merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
    
    nx.write_gpickle(merge_contract_graph, join(output, file_name_sc))
    return 1

from pathlib import Path
def auto_generate_dirs(path):
    """
    Automatically generate directories based on the given path.
    If the path ends with a file extension, use the parent directory instead.
    
    Args:
        path (str): The path for which to create directories.
    """
    # Convert to Path object for easier manipulation
    path_obj = Path(path)
    
    # Check if the path appears to end with a file extension
    # (has a suffix and the suffix is not just a dot)
    if path_obj.suffix and path_obj.suffix != '.':
        # Use the parent directory instead
        dir_path = path_obj.parent
    else:
        dir_path = path_obj
    
    try:
        # Create directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"{Back.GREEN}Successfully created directory: {dir_path}{Style.RESET_ALL}")
    except OSError as e:
        print(f"{Back.RED}Error creating directory {dir_path}: {e}{Style.RESET_ALL}")


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
    print("Usage: python control_flow_graph_generator.py")
    print(f"{Fore.CYAN} This script generates control flow graphs (CFG) for Solidity smart contracts.{Style.RESET_ALL}")
    print("It processes contracts in the specified directories, creates CFGs,")
    print("and outputs compressed graph files in .gpickle format.")
    print("\nThe script uses predefined paths and configurations. To modify:\n")
    print("1. Update the 'ROOT' variable to change the base directory.")
    print("2. Modify the 'bug_type' dictionary to adjust contract categories and counts.")
    print("3. Ensure vulnerability JSON files are in the correct locations.")
    print(f"\n{Fore.YELLOW}No command-line arguments are required. Modify the script to change paths and bug types.{Style.RESET_ALL}")
    # Function to wait for user input before proceeding
    print(f"\n [1] Generates original")
    print(f"\n [2] Generates from new")
    print(f"\n\n {Back.RED}!!! Warning: This script may install many versions of Solc. Recommended to use Docker or venv.{Style.RESET_ALL}\n\n")

    o = input(f"{Back.BLUE}Press 1 OR 2 to start generating control flow graphs...{Style.RESET_ALL}")
    if o not in ['1','2']:
        print(f"{Back.RED}Invalid input! Exiting...{Style.RESET_ALL}")
        sys.exit(1)
    return o




if __name__ == '__main__':
    
    # smart_contract_path = './data/clean_71_buggy_curated_0'
    # input_graph = None
    # output_path = './data/clean_71_buggy_curated_0/cfg_compress_graphs.gpickle'
    # smart_contracts = [join(smart_contract_path, f) for f in os.listdir(smart_contract_path) if f.endswith('.sol')]

    # data_vulnerabilities = None
    # list_vulnerabilities_json_files = [
    #     './data/solidifi_buggy_contracts/reentrancy/vulnerabilities.json',
    #     # 'data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
    #     './data/smartbug-dataset/vulnerabilities.json']
    
    # data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
    
    # compress_full_smart_contracts(smart_contracts, input_graph, output_path, vulnerabilities=data_vulnerabilities)

    ROOT = './experiments/ge-sc-data/source_code'
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    option = print_info()
    if option =='1':
        print(f"{Back.CYAN}ROOT: {ROOT}{Style.RESET_ALL}")
        print(f"{Back.CYAN}Starting control flow graph generation...{Style.RESET_ALL}")
        for bug, counter in bug_type.items():
            # source = f'{ROOT}/{bug}/buggy_curated'
            # output = f'{ROOT}/{bug}/buggy_curated/cfg_compressed_graphs.gpickle'
            print(f"{Fore.CYAN}Processing bug type: {bug}{Style.RESET_ALL}")
            #############################  
            # Change here!
            source = f'{ROOT}/{bug}/curated'
            specific_name = "" # Leave empty for default
            output_CFG_path = f'{ROOT}/{specific_name}{bug}/curated/cfg_compressed_graphs.gpickle'
            print(f"{Back.CYAN}OUTPUT: {output_CFG_path}{Style.RESET_ALL}")
            #############################

            auto_generate_dirs(output_CFG_path)
            smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]
            data_vulnerabilities = None
            list_vulnerabilities_json_files = ['data/solidifi_buggy_contracts/reentrancy/vulnerabilities.json',
            # 'data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
            'data/smartbug-dataset/vulnerabilities.json']
            data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
            compress_full_smart_contracts(smart_contracts, None, output_CFG_path, vulnerabilities=data_vulnerabilities)
    elif option == '2':
        output_CFG_path = f'./newMethods/sampleDataset/cfg_compressed_graphs.gpickle'
        source = f'./newMethods/sampleDataset/'
        n = 10  # Replace with the number of contracts you want
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]#[:n]
        compress_full_smart_contracts(smart_contracts, None, output_CFG_path, vulnerabilities=None)
    print(f"{Back.GREEN}Control flow graph generation completed!{Style.RESET_ALL}")