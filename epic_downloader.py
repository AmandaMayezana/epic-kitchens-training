# epic_downloader.py - MODIFIED FOR SUBSET DOWNLOAD
import os
import requests
import zipfile
import argparse

def download_subset(dataset_url, output_dir, subset_size=10):
    """
    Download only a manageable subset of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Download only specific classes or time segments
    # Focus on 2-3 intention classes with limited samples
    print(f"Downloading subset to {output_dir}")
    
    # Your implementation here for selective download
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data_subset")
    parser.add_argument("--subset_size", type=int, default=10)
    args = parser.parse_args()
    
    download_subset(
        dataset_url="https://mobility.iiit.ac.in/icpr_2024_rip",
        output_dir=args.output_dir,
        subset_size=args.subset_size
    )
