import os
import json
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import itertools

import colorama
from colorama import Fore, Style, Back
colorama.just_fix_windows_console()



# Mapping for category names to CSV columns
CATEGORY_MAPPING = {
    "arithmetic": ["Underflow", "Overflow"],
    "reentrancy": ["Reentracy"], #yep this is a typo
    "time_manipulation": ["TOD", "TimeDep", "BlockTimestamp"],
    "unchecked_low_level_calls": ["LowlevelCalls"],
    "access_control": ["SelfDestruct"],
    "denial_of_service": ["CallDepth"],
    #"front_running": [],
    #"other": ["AssertFail", "CheckEffects", "InlineAssembly"]
}


def collect_json_files(directory):
    """Recursively collect all JSON files in a directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file=="test_report.json":
                json_files.append(os.path.join(root, file))
    return json_files

def parse_test_report(file_paths):
    """
    Extract vuln for each contract from test_report.json.
    """
    vuln = {}
    raw_data = {}
    
    for file_path in file_paths:
        print(f"Processing {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            category = os.path.basename(os.path.dirname(file_path))
            print("Processing category:", category)
        
            for reports in data:
                processed_contracts = 0
                for idx, contract in enumerate(reports):
                    
                    contract_base_name = contract.split('-')[0] # Remove fuction name
                    # Initialize if contract not in vuln
                    if contract_base_name not in vuln:
                        vuln[contract_base_name] = {category: "" for category in CATEGORY_MAPPING.keys()} # Init to match result columns
                        processed_contracts += 1

                    if contract_base_name not in raw_data:
                        raw_data[contract_base_name] = {category: "" for category in CATEGORY_MAPPING.keys()}

                    # Map to individual categories
                    if category in CATEGORY_MAPPING:
                        vuln[contract_base_name][category] = 1 if reports[contract][1] > 0.5 else 0
                        raw_data[contract_base_name][category] = reports[contract]
                        
                print(f"Processed new {processed_contracts} contracts")  

        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")

    return vuln, raw_data

def load_csv(csv_path):
    """Load the CSV file into a pandas DataFrame and process it."""
    try:
        df = pd.read_csv(csv_path)
        df.drop('source_code', axis=1, inplace=True)
        csv_data = {row['Addr']: row.drop('Addr').values.tolist() for _, row in df.iterrows()}
        print(dict(itertools.islice(csv_data.items(), 10)))
        return csv_data
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return {}

def calculate_label_stats(data_dict, category_mapping):
    stats = {category: 0 for category in category_mapping.keys()}
    for values in data_dict.values():
        for idx, val in enumerate(values):
            if val == 1:
                stats[list(category_mapping.keys())[idx]] += 1
    return stats

def compare_results(json_vuln, csv_data, save_dir):
    """
    Compare JSON vuln with CSV data and calculate accuracy, precision, and F1 score 
    for each vulnerability category.
    """
    # Prepare lists to store results for each category
    category_results = {
        category: {
            'actuals': [],
            'predictions': []
        } for category in CATEGORY_MAPPING.keys()
    }
    
    # Get the column names from your DataFrame
    column_names = ["Underflow", "Overflow", "CallDepth", "TOD", "TimeDep", 
                    "Reentracy", "AssertFail", "CheckEffects", "InlineAssembly", 
                    "BlockTimestamp", "LowlevelCalls", "SelfDestruct"]

    processed_data = {}
    found_contracts_categories = {
        "actuals": {subcat: 0 for subcat in column_names},
        "remapped": { k: 0 for k in CATEGORY_MAPPING.keys()},
    }

    actual_remapped = {}

    for addr, values in csv_data.items():
        # Skip if address not in json_vuln
        if addr not in json_vuln.keys():
            continue
        
        # Create a dictionary from the values list
        row_dict = dict(zip(column_names, values))
        actual_remapped[addr] = {category: 0 for category in CATEGORY_MAPPING.keys()}
        #print('================================================================================================')
        # Iterate through each vulnerability category
        for cat_idx, (category, sub_categories) in enumerate(CATEGORY_MAPPING.items()):
            # If no subcategories defined for the category, skip
            if not sub_categories:
                continue
            # total number of actual categories
            for subcat in sub_categories:
                found_contracts_categories["actuals"][subcat] += int(row_dict.get(subcat, 0))
            # Check each subcategory for the current address
            actuals_category_total = 1 if sum(int(row_dict.get(subcat, 0)) for subcat in sub_categories) >= 1 else 0
            
            actual_remapped[addr][category] = actuals_category_total
            if actuals_category_total == 1:
                found_contracts_categories["remapped"][category] += 1
            #print(f"{Fore.GREEN}Address: {addr}, Category: {category}, Actual: {actuals_category_total}, Predicted: {json_vuln[addr][cat_idx]}{Style.RESET_ALL}")
            # Store results for each category
            category_results[category]['actuals'].append(actuals_category_total)
            category_results[category]['predictions'].append(json_vuln[addr][category])
        # Store the processed data
        processed_data[addr] = actual_remapped[addr]
        #print(f"{Fore.GREEN}Address: {addr}, Actual: {actual_remapped}, Predicted: {json_vuln[addr]} {Style.RESET_ALL} ")
        #print('================================================================================================\n\n')

    with open(f'{save_dir}actual_remapped.json', 'w') as f:
        json.dump(actual_remapped, f)

    category_metrics = {}
    for category, data in category_results.items():
        #f category == 'reentrancy':
          #  print(data)
        if not data['actuals']:
            category_metrics[category] = {
                'accuracy': 0,
                'precision': 0,
                'f1': 0,
                'confusion_matrix': None
            }
        else:
            cfm = confusion_matrix(data['actuals'], data['predictions'],labels=[0,1])
            
            category_metrics[category] = {
                'accuracy': accuracy_score(data['actuals'], data['predictions']),
                'precision': precision_score(data['actuals'], data['predictions'],labels=[0,1]),
                'f1': f1_score(data['actuals'], data['predictions']),
                'cfm': cfm
            }


    # Print summary
    print("\nLabel Distribution:")
    # csv_stats = calculate_label_stats(processed_data, CATEGORY_MAPPING)
    for category in found_contracts_categories['actuals'].keys():
        print(f"{category}: {found_contracts_categories['actuals'][category]}")

    print("\nRemapped Label Distribution:")
    for category in found_contracts_categories['remapped'].keys():
        print(f"{category}: {found_contracts_categories['remapped'][category]}")


    # Print metrics
    print("\nCategory-wise Metrics:")
    for category, metrics in category_metrics.items():
        print(f"\n{category}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'cfm' in metrics:
            metrics['tn'],metrics['fp'], metrics['fn'], metrics['tp'] = metrics['cfm'].ravel()
            print(f"  False Positives: {metrics['fp']}")
            print(f"  False Negatives: {metrics['fn']}")
            print(f"  True Positives: {metrics['tp']}")
            print(f"  True Negatives: {metrics['tn']}")
            cfmD = ConfusionMatrixDisplay(confusion_matrix=metrics['cfm'], display_labels = [0, 1])
            cfmD.plot()
            cfmD.ax_.set_title(f'{category} Confusion Matrix')
            os.makedirs("./newMethods/sampleDataset_fig/", exist_ok=True)
            cfmD.figure_.savefig(f'./newMethods/sampleDataset_fig/{category}_cfm.png')



def generate_report(output_path, accuracy, precision, f1):
    """Write the accuracy, precision, and F1 score to the output file."""
    try:
        with open(output_path, 'w') as f:
            f.write("Test Report:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print(f"Report written to {output_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")

def main():
    print("EXAMPLE: python newMethods/result_parse.py -d newMethods/logs/test_logs/2024-12-27_14-37-17/ -c newMethods/archive/SoliAudit-VA-Dataset-SourceCode.csv")
    parser = argparse.ArgumentParser(description="Test Report Comparator with Mapping \n\n EX: \npython newMethods/result_parse.py -d newMethods/logs/test_logs/2024-12-27_14-37-17/ -c newMethods/archive/SoliAudit-VA-Dataset-SourceCode.csv")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing JSON files")
    parser.add_argument("-c", "--csv", required=True, help="Path to CSV file")
    args = parser.parse_args()

    # Step 1: Collect JSON files
    json_files = collect_json_files(args.directory)
    print(f"Found {len(json_files)} JSON files.")

    # Step 2: Parse all test_report.json files
    json_vuln = {}
    raw_vuln = {}
    json_vuln, raw_vuln = parse_test_report(json_files)


    #Save json
    with open(f'{args.directory}test_report_combined.json', 'w') as f:
        json.dump(json_vuln, f)

    with open(f'{args.directory}test_report_raw.json', 'w') as f:
        json.dump(raw_vuln, f)
    

    # Step 3: Load CSV data
    csv_data = load_csv(args.csv)
    if not csv_data:
        print("No valid CSV data found. Exiting...")
        return

    # Step 4: Compare and calculate metrics
    compare_results(json_vuln, csv_data, args.directory)


main()
