import torch
import sys
import os
import pandas as pd
import numpy as np

# ==============================================================================
# Step 1: Fix Python's Path to find project modules
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
print(f"Adding project root to Python path: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)

# ==============================================================================
# Step 2: Import the target module directly
# ==============================================================================
try:
    from models.backbone_module import Pointnet2Backbone
except ImportError as e:
    print("\n--- IMPORT ERROR ---")
    print(f"Failed to import Pointnet2Backbone: {e}")
    print("This is expected if you have not compiled the pointnet2 extensions.")
    print("Please navigate to the 'pointnet2' directory, fix any C++ errors, and run: python setup.py install")
    sys.exit()

# ==============================================================================
# Step 3: Standalone Test Execution
# ==============================================================================

def load_point_cloud_from_csv(filename, feature_columns=['INTENSITY']):
    """
    Loads a point cloud from a CSV file and returns a tensor.
    """
    print(f"Loading point cloud from {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The specified CSV file was not found: {filename}")
        
    df = pd.read_csv(filename)
    required_cols = ['X', 'Y', 'Z'] + feature_columns
    for col in required_cols:
        if col not in df.columns: raise ValueError(f"CSV missing required column: {col}")
            
    coords = df[['X', 'Y', 'Z']].values
    features = df[feature_columns].values
    
    point_cloud_np = np.hstack([coords, features]).astype(np.float32)
    return torch.from_numpy(point_cloud_np).unsqueeze(0)


if __name__ == '__main__':
    print("\n--- Testing Pointnet2Backbone (Real Implementation) ---")

    # --- Configuration ---
    INPUT_FEATURE_DIM = 1 # For the 'INTENSITY' column
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This script will fail because the compiled code requires a GPU.")


    # --- Create Model and move it to the correct device ---
    model = Pointnet2Backbone(input_feature_dim=INPUT_FEATURE_DIM)
    # THE FIX: Move the model to the GPU.
    model.to(DEVICE)
    print(f"Successfully instantiated Pointnet2Backbone and moved to {DEVICE}")
    

    # --- Load Data from your CSV and move it to the correct device ---
    try:
        csv_file_sample = r"/media/khaled/Ananas/Recorded_Data_BZ1_BZ2/csv_Data/Data1_croped/X_00001.csv"
        input_tensor = load_point_cloud_from_csv(csv_file_sample, feature_columns=['INTENSITY'])
        
        # THE FIX: Move the input data to the GPU.
        input_tensor = input_tensor.to(DEVICE)
            
        print(f"Loaded data from CSV. Input tensor shape: {input_tensor.shape}, Device: {input_tensor.device}")
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        sys.exit()
    
    print("-" * 50)
    print("Executing model.forward()...\n")
    
    # Run the forward pass with the real, compiled modules
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        # Note: The original forward pass takes an optional end_points dict.
        end_points = model(input_tensor, {})
    
    print("\n...forward() pass completed.")
    print("-" * 50)

    # --- Inspect the Output ---
    print("\n--- Final Output Tensor Shapes ('end_points') ---")
    for key, value in end_points.items():
        # Move tensor to CPU for printing, to avoid potential CUDA errors
        print(f"  '{key}':".ljust(20) + f"Shape: {value.cpu().shape}")
    
    print("\nTest finished. This shows the data flow using the real, compiled modules.")
