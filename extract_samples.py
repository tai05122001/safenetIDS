import pandas as pd
import os

input_pkl_path = "C:/gitlab/safenetIDS/safenetIDS/dataset/splits/level1/test.pkl"
output_csv_path = "C:/gitlab/safenetIDS/safenetIDS/services/data/test_samples.csv"

# Ensure the output directory exists
output_dir = os.path.dirname(output_csv_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # Read the pickle file
    df = pd.read_pickle(input_pkl_path)
    
    # Separate benign and attack samples
    benign_samples = df[df['label'] == 'Benign']
    attack_samples = df[df['label'] != 'Benign']
    
    # Take 5 benign and 5 attack samples, or fewer if not enough exist
    selected_samples = []
    if len(benign_samples) >= 5:
        selected_samples.append(benign_samples.sample(n=5, random_state=42))
    else:
        selected_samples.append(benign_samples)
        
    if len(attack_samples) >= 5:
        selected_samples.append(attack_samples.sample(n=5, random_state=42))
    else:
        selected_samples.append(attack_samples)

    # Concatenate the selected samples
    final_df = pd.concat(selected_samples).sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    
    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"Successfully extracted {len(final_df)} samples to {output_csv_path}")

except FileNotFoundError:
    print(f"Error: The file {input_pkl_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
