import pandas as pd
import os
import argparse

def parse_csv_to_sol(input_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # Check if 'sourcecode' column exists
    if 'source_code' not in df.columns:
        print("Error: 'sourcecode' column not found in the CSV file.")
        return

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the sourcecode content
        sourcecode = row['source_code']
        basename = row['Addr']
        
        # Generate a unique filename for each .sol file
        filename = f"{basename}.sol"  # +2 to account for 0-indexing and header row
        filepath = os.path.join(output_dir, filename)

        # Write the sourcecode to a .sol file
        with open(filepath, 'w', encoding='utf-8') as solfile:
            solfile.write(sourcecode)

        print(f"Created: {filepath}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse 'sourcecode' column from CSV to .sol files")
    parser.add_argument('-i', '--input', required=True, help="Input CSV file path")
    parser.add_argument('-o', '--output', required=True, help="Output directory for .sol files")

    # Parse arguments
    args = parser.parse_args()

    # Call the function to parse CSV and create .sol files
    parse_csv_to_sol(args.input, args.output)

if __name__ == "__main__":
    main()