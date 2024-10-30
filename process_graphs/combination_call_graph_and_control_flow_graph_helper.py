import networkx as nx
from os.path import join
import colorama
from colorama import Fore, Style, Back
# Initialize colorama for Windows compatibility
colorama.just_fix_windows_console()

def print_nx_network_full_info(nx_graph):
    print('====Nodes info====')
    for node, node_data in nx_graph.nodes(data=True):
        print(node, node_data)
    
    # print('====Edges info====')
    # for source_node, target_node, edge_data in nx_graph.edges(data=True):
    #     print(source_node, target_node, edge_data)

def mapping_cfg_and_cg_node_labels(cfg, call_graph):
    dict_node_label_cfg_and_cg = {}

    for node, node_data in cfg.nodes(data=True):
        if node_data['node_type'] == 'FUNCTION_NAME':
            if node_data['label'] not in dict_node_label_cfg_and_cg:
                dict_node_label_cfg_and_cg[node_data['label']] = None
            # else:
            #     print(node_data['label'], 'is existing.')

            dict_node_label_cfg_and_cg[node_data['label']] = {
                'cfg_node_id': node,
                'cfg_node_type': node_data['node_type']
            }
    
    
    for node, node_data in call_graph.nodes(data=True):
        if node_data['label'] in dict_node_label_cfg_and_cg:
            dict_node_label_cfg_and_cg[node_data['label']]['call_graph_node_id'] = node
            dict_node_label_cfg_and_cg[node_data['label']]['call_graph_node_type'] = node_data['node_type'].upper()
        else:
            print(node_data['label'], ' is not existing.')


    # remove node labels are not existing in the call graph
    temp_dict = dict(dict_node_label_cfg_and_cg)
    for key, value in temp_dict.items():
        if 'call_graph_node_id' not in value or 'call_graph_node_type' not in value:
            dict_node_label_cfg_and_cg.pop(key, None)

    return dict_node_label_cfg_and_cg

def add_new_cfg_edges_from_call_graph(cfg, dict_node_label, call_graph):
    list_new_edges_cfg = []
    for source, target, edge_data in call_graph.edges(data=True):
        source_cfg = None
        target_cfg = None
        edge_data_cfg = edge_data
        for value in dict_node_label.values():
            if value['call_graph_node_id'] == source:
                source_cfg = value['cfg_node_id']
            
            if value['call_graph_node_id'] == target:
                target_cfg = value['cfg_node_id']
        
        if source_cfg is not None and target_cfg is not None:
            list_new_edges_cfg.append((source_cfg, target_cfg, edge_data_cfg))
    
    cfg.add_edges_from(list_new_edges_cfg)

    return cfg
    
def update_cfg_node_types_by_call_graph_node_types(cfg, dict_node_label):
    for value in dict_node_label.values():
        cfg_node_id = value['cfg_node_id']
        cfg.nodes[cfg_node_id]['node_type'] = value['call_graph_node_type']


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
    print("Usage:")
    print(f"{Fore.CYAN}This script combines Control Flow Graphs (CFG) and Call Graphs (CG) for Solidity smart contracts.{Style.RESET_ALL}")
    print("\nFunctionality:\n")
    print("1. Reads CFG and CG from input .gpickle files")
    print("2. Maps nodes between CFG and CG")
    print("3. Adds new edges to CFG based on CG")
    print("4. Updates CFG node types using CG node types")
    print("5. Outputs a merged graph as both .dot and .gpickle files")
    print("\n\nMain operations:\n")
    print("- Processes multiple bug types (access_control, arithmetic, etc.)")
    print("- Generates combined CFG-CG graphs for each bug type")
    print("- Saves the results in the specified output directories")
    print(f"\n{Fore.YELLOW}No command-line arguments are required. Modify the script to change paths and bug types.{Style.RESET_ALL}")

    input(f"{Back.BLUE}Press Enter to start...{Style.RESET_ALL}")

if __name__ == '__main__':
    # input_cfg_path = 'data/clean_57_buggy_curated_0_access_control/cfg_compress_graphs.gpickle'
    # input_call_graph_path = 'data/clean_57_buggy_curated_0_access_control/compress_call_graphs_no_solidity_calls_buggy.gpickle'
    # output_path = 'data/clean_57_buggy_curated_0_access_control'

    # input_cfg = nx.read_gpickle(input_cfg_path)
    # print(nx.info(input_cfg))
    # # print_nx_network_full_info(input_cfg)

    # input_call_graph = nx.read_gpickle(input_call_graph_path)
    # print(nx.info(input_call_graph))
    # # print_nx_network_full_info(input_call_graph)

    # dict_node_label_cfg_and_cg = mapping_cfg_and_cg_node_labels(input_cfg, input_call_graph)

    # merged_graph = add_new_cfg_edges_from_call_graph(input_cfg, dict_node_label_cfg_and_cg, input_call_graph)
    # # print(nx.info(merged_graph))
    # # print_nx_network_full_info(input_cfg)

    # update_cfg_node_types_by_call_graph_node_types(merged_graph, dict_node_label_cfg_and_cg)
    # print(nx.info(merged_graph))
    # # print_nx_network_full_info(input_cfg)

    # nx.nx_agraph.write_dot(merged_graph, join(output_path, 'merged_graph_cfg_and_cg.dot'))
    # print('Dumped succesfully:', join(output_path, 'merged_graph_cfg_and_cg.dot'))
    # nx.write_gpickle(merged_graph, join(output_path, 'merged_graph_cfg_and_cg.gpickle'))
    # print('Dumped succesfully:', join(output_path, 'merged_graph_cfg_and_cg.gpickle'))

    ROOT = './experiments/ge-sc-data/source_code'
    bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
              'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
              'unchecked_low_level_calls': 95}
    
    print_info()
    print(f"{Back.CYAN} ROOT: {ROOT}{Style.RESET_ALL}")
    print(f"{Back.CYAN}Starting to process combine...{Style.RESET_ALL}")

    for bug, counter in bug_type.items():
        print(f"{Back.CYAN}Starting to process {str(bug)}bug...{Style.RESET_ALL}")
        #############################  
        # Change here!
        source = f'{ROOT}/{bug}/curated'
        specific_name = "Null/" # Leave empty for default
        output_CFGxCG_path = f'{ROOT}/{specific_name}{bug}/curated/cfg_cg_compressed_graphs.gpickle'
        
        input_cfg_path = f'{ROOT}/{specific_name}{bug}/curated/cfg_compressed_graphs.gpickle'
        input_call_graph_path = f'{ROOT}/{specific_name}{bug}/curated/cg_compressed_graphs.gpickle'

        print(f"{Back.CYAN}OUTPUT: {output_CFGxCG_path}{Style.RESET_ALL}")
        print(f"{Back.CYAN}INPUT CFG: {input_cfg_path}{Style.RESET_ALL}")
        print(f"{Back.CYAN}INPUT CG: {input_call_graph_path}{Style.RESET_ALL}")
        #############################  
        input_cfg = nx.read_gpickle(input_cfg_path)
        input_call_graph = nx.read_gpickle(input_call_graph_path)
        dict_node_label_cfg_and_cg = mapping_cfg_and_cg_node_labels(input_cfg, input_call_graph)
        merged_graph = add_new_cfg_edges_from_call_graph(input_cfg, dict_node_label_cfg_and_cg, input_call_graph)
       
        # output_CFGxCG_path = f'/home/minhnn/minhnn/ICSE/ge-sc/experiments/ge-sc-data/source_code/compressed_graphs/buggy_curated/{bug}_cfg_cg_compressed_graphs.gpickle'
        update_cfg_node_types_by_call_graph_node_types(merged_graph, dict_node_label_cfg_and_cg)
        nx.write_gpickle(merged_graph, output_CFGxCG_path)
        print(f"{Fore.GREEN}Dumped succesfully: {output_CFGxCG_path}{Style.RESET_ALL}")

    print(f"{Back.GREEN}Process completed!{Style.RESET_ALL}")
    # # Combine all the graphs
    # multi_graphs = [nx.read_gpickle(f'{ROOT}/{bug}/buggy_curated/cfg_cg_compressed_graphs.gpickle') for bug in bug_type.keys()]
    # for bug, counter in bug_type.items():
    #     source_graph = f'{ROOT}/{bug}/buggy_curated/cfg_cg_compressed_graphs.gpickle'
    #     full_graph = nx.disjoint_union_all(multi_graphs)
    # nx.write_gpickle(full_graph, f'{ROOT}/compressed_graphs/multi_bug_cfg_cg_compressed_graphs.gpickle')