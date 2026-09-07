import os
import sys

# Ensure local imports work correctly for the Model file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.multimodal_model import EAVMultimodalFusionNetwork

def main():
    print("Loading EAV Multimodal Architecture...")
    
    # Instantiate the exact binary configuration model
    model = EAVMultimodalFusionNetwork(num_classes=2)
    
    # Create the docs directory if it doesn't already exist
    os.makedirs('docs', exist_ok=True)
    save_path = 'docs/Model_Topology.txt'
    
    # We attempt to print via torchinfo for extremely deep parameter tracking.
    # Otherwise, we default to the standard PyTorch object __repr__.
    
    output = ""
    try:
        import torchinfo
        print("torchinfo module detected! Generating explicit parameter list...")
        
        # Give torchinfo dummy tensors exactly matching the input shapes
        # eeg: (B, 30, 500)
        # aud: (B, 80000)
        # vis: (B, 25, 3, 56, 56)
        summary = torchinfo.summary(
            model, 
            input_size=[(1, 30, 500), (1, 80000), (1, 25, 3, 56, 56)], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
            verbose=0
        )
        output = str(summary)
        
    except ImportError:
        print("Note: 'torchinfo' not installed. Defaulting to standard PyTorch format.")
        print("To see exact parameter counts in the future, run: pip install torchinfo")
        output = str(model)

    # Save output to text file
    with open(save_path, 'w') as f:
        f.write(output)
        
    print(f"\nArchitecture successfully analyzed and saved to: {save_path}")

if __name__ == "__main__":
    main()
