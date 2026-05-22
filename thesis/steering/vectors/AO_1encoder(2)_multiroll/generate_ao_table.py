import pandas as pd

def main():
    input_path = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll/multiroll_ao_indices.csv'
    output_path = '/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_multiroll/ao_corrected_table.csv'
    
    # Read the updated CSV
    df = pd.read_csv(input_path)
    
    # Extract the numeric step value from the 'Folder' column (e.g., 'steps_12' -> 12)
    df['Step'] = df['Folder'].str.extract(r'steps_(\d+)').astype(int)
    
    # Create the 2D matrix mapping Alpha (rows) to Steps (columns) using the Corrected AO index
    pivot_df = df.pivot(index='Alpha', columns='Step', values='AO_Index_Corrected')
    
    # Ensure rows and columns are sorted numerically
    pivot_df = pivot_df.sort_index()
    pivot_df = pivot_df.sort_index(axis=1)
    
    # Set the index name so the first column header makes sense
    pivot_df.index.name = 'Alphas'
    
    # Save the updated table to CSV
    pivot_df.to_csv(output_path)
    print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    main()
