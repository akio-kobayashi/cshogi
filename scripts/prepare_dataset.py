"""
This script prepares a dataset for training by processing multiple CSA files in parallel.

It reads a metadata CSV file to get the paths to CSA files, then uses the
UnifiedFeatureGenerator to extract features from each game. The results are
saved to an output directory.
"""
import os
import json
import argparse
import multiprocessing
import pandas as pd
from functools import partial

# Add the project root to the path to allow importing cshogi
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cshogi
from cshogi.dlshogi.dataprep import UnifiedFeatureGenerator

# --- Global objects for multiprocessing ---
# These are initialized once per process in the pool initializer.
generator = None

def init_worker(config_path):
    """
    Initializer for each worker in the multiprocessing pool.
    Loads the feature configuration and creates a UnifiedFeatureGenerator instance.
    """
    global generator
    print(f"Initializing worker {os.getpid()}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    generator = UnifiedFeatureGenerator(config=config)

def process_csa_worker(csa_file_path, output_dir, perspective, sampling_options):
    """
    A worker function that processes a single CSA file.

    It extracts features using the global generator instance and saves the
    result as a compressed NumPy file (.npz).
    """
    try:
        # Generate a unique output filename based on the CSA file name
        base_name = os.path.splitext(os.path.basename(csa_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.npz")

        if os.path.exists(output_path):
            print(f"Skipping already processed file: {csa_file_path}")
            return None

        # Use the global generator instance initialized for this process
        result_dict = generator.process_csa(
            csa_file_path,
            perspective,
            sampling_options
        )

        # Save the features, SFENs, and result to a compressed .npz file
        np.savez_compressed(
            output_path,
            features=result_dict["features"],
            sfens=result_dict["sfens"],
            result=np.array(result_dict["result"]) # npz works best with arrays
        )

        return f"Successfully processed {csa_file_path}"

    except Exception as e:
        return f"Error processing {csa_file_path}: {e}"

def main():
    """
    Main function to orchestrate the parallel dataset preparation.
    """
    parser = argparse.ArgumentParser(
        description="Parallel processing of CSA files to generate features for ML training."
    )
    parser.add_argument("metadata_csv", type=str, help="Path to the metadata.csv file.")
    parser.add_argument("config_json", type=str, help="Path to the feature_config.json file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed .npz files.")
    parser.add_argument(
        "--perspective", type=str, default="black", choices=["black", "white"],
        help="The perspective for feature extraction ('black' or 'white'). Default: black."
    )
    parser.add_argument(
        "--sampling_method", type=str, default="interval", choices=["none", "interval", "random"],
        help="Sampling method to use. Default: interval."
    )
    parser.add_argument("--sampling_n", type=int, default=5, help="Interval for 'interval' sampling.")
    parser.add_argument("--sampling_k", type=int, default=20, help="Number of samples for 'random' sampling.")
    parser.add_argument(
        "--num_workers", type=int, default=multiprocessing.cpu_count(),
        help="Number of worker processes to use. Default: all available CPU cores."
    )
    
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the list of CSA files to process
    try:
        metadata_df = pd.read_csv(args.metadata_csv)
        csa_files = metadata_df['file_path'].tolist()
        print(f"Found {len(csa_files)} CSA files to process from {args.metadata_csv}.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {args.metadata_csv}")
        return
    except KeyError:
        print(f"Error: 'file_path' column not found in {args.metadata_csv}")
        return

    # Prepare sampling options dictionary
    sampling_options = None
    if args.sampling_method == 'interval':
        sampling_options = {'method': 'interval', 'n': args.sampling_n}
    elif args.sampling_method == 'random':
        sampling_options = {'method': 'random', 'k': args.sampling_k}

    # Set perspective
    perspective_val = cshogi.BLACK if args.perspective == 'black' else cshogi.WHITE
    
    # --- Parallel Processing ---
    # Use functools.partial to create a version of the worker function with fixed arguments
    worker_func = partial(
        process_csa_worker,
        output_dir=args.output_dir,
        perspective=perspective_val,
        sampling_options=sampling_options
    )

    print(f"Starting parallel processing with {args.num_workers} workers...")

    # Create a pool of worker processes
    # The `initializer` and `initargs` ensure that each worker process
    # gets its own `UnifiedFeatureGenerator` instance without needing to
    # create it for every single task.
    with multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args.config_json,)) as pool:
        # Use imap_unordered to process files and get results as they complete
        results_iterator = pool.imap_unordered(worker_func, csa_files)
        
        # Process results and show progress
        for i, result in enumerate(results_iterator):
            if result:
                print(f"[{i+1}/{len(csa_files)}] {result}")

    print("--- Dataset preparation complete. ---")
    print(f"Processed files are saved in: {args.output_dir}")

if __name__ == "__main__":
    # To run this script:
    # 1. Create a metadata.csv file with a 'file_path' column pointing to your CSA files.
    # 2. Create a feature_config.json file.
    # 3. Run from the command line:
    #    python scripts/prepare_dataset.py /path/to/metadata.csv /path/to/feature_config.json /path/to/output_dir
    main()
