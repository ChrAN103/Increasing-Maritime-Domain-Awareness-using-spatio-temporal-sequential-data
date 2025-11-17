"""
Simple script to download parquet files from Google Drive and convert to pandas DataFrame.

Usage:
    python download_parquet.py
"""

from google_drive_fetcher import GoogleDriveFetcher
import pandas as pd


def download_parquet_as_dataframe(parent_folder_url, max_depth=5, num_files=1, combine=True):
    """
    Download parquet file(s) from Google Drive and return as DataFrame.
    
    Args:
        parent_folder_url: URL of the parent Google Drive folder
        max_depth: Maximum depth to search for parquet files
        num_files: Number of files to download (default: 1)
        combine: If True and num_files > 1, combine into single DataFrame (default: True)
        
    Returns:
        pandas DataFrame (if combine=True or num_files=1) or list of DataFrames (if combine=False)
        Returns None if no parquet files found
    """
    print("Initializing Google Drive Fetcher...")
    fetcher = GoogleDriveFetcher()
    
    print(f"Searching for {num_files} parquet file(s) (max depth: {max_depth})...\n")
    
    parquet_files = _find_parquet_files(fetcher, parent_folder_url, max_depth, num_files)
    
    if parquet_files:
        print(f"\n✓ Found {len(parquet_files)} file(s)")
        
        dataframes = []
        for i, parquet_file in enumerate(parquet_files, 1):
            print(f"\n[{i}/{len(parquet_files)}] Downloading: {parquet_file['name']}")
            df = fetcher.fetch_parquet_to_dataframe(parquet_file['url'])
            print(f"  Shape: {df.shape}")
            dataframes.append(df)
        
        if num_files == 1 or combine:
            if len(dataframes) > 1:
                print(f"\nCombining {len(dataframes)} DataFrames...")
                combined_df = pd.concat(dataframes, ignore_index=True)
                print(f"✓ Combined DataFrame shape: {combined_df.shape}")
                print(f"Columns: {list(combined_df.columns)}")
                return combined_df
            else:
                print(f"\n✓ Success!")
                print(f"DataFrame shape: {dataframes[0].shape}")
                print(f"Columns: {list(dataframes[0].columns)}")
                return dataframes[0]
        else:
            print(f"\n✓ Success! Returning {len(dataframes)} separate DataFrames")
            return dataframes
    else:
        print("\n❌ No parquet files found")
        return None


def _find_parquet_files(fetcher, folder_url, max_depth=5, max_files=5, current_depth=0, found_files=None):
    """
    Recursively search for multiple parquet files.
    
    Args:
        fetcher: GoogleDriveFetcher instance
        folder_url: Google Drive folder URL
        max_depth: Maximum recursion depth
        max_files: Maximum number of files to find
        current_depth: Current depth level
        found_files: List to collect found files
        
    Returns:
        List of dicts with file info
    """
    if found_files is None:
        found_files = []
    
    # Stop if we've found enough files or reached max depth
    if len(found_files) >= max_files or current_depth >= max_depth:
        return found_files
    
    indent = "  " * current_depth
    print(f"{indent}📂 Level {current_depth}: Checking folder...")
    
    try:
        contents = fetcher.list_directory(folder_url)
        
        # First pass: Look for parquet files
        for item in contents:
            if item['type'] == 'file' and '.parquet' in item['name'] and 'Shared folder' not in item['name']:
                if len(found_files) < max_files:
                    print(f"{indent}  ✓ Found: {item['name']}")
                    found_files.append(item)
                    if len(found_files) >= max_files:
                        return found_files
        
        # Second pass: Check subfolders if we need more files
        if len(found_files) < max_files:
            folders = [item for item in contents if item['type'] == 'folder' or 'Shared folder' in item['name']]
            
            for folder in folders:
                if len(found_files) >= max_files:
                    break
                    
                # Convert to folder URL if needed
                folder_url_converted = folder['url']
                if '/file/d/' in folder_url_converted:
                    folder_url_converted = f"https://drive.google.com/drive/folders/{folder['id']}"
                
                _find_parquet_files(fetcher, folder_url_converted, max_depth, max_files, current_depth + 1, found_files)
        
        return found_files
        
    except Exception as e:
        print(f"{indent}❌ Error: {e}")
        return found_files


def _find_first_parquet(fetcher, folder_url, max_depth=5, current_depth=0):
    """
    Recursively search for the first parquet file.
    (Kept for backward compatibility)
    """
    files = _find_parquet_files(fetcher, folder_url, max_depth, max_files=1, current_depth=current_depth)
    return files[0] if files else None


if __name__ == "__main__":
    # Example: Download from maritime AIS data folder
    PARENT_FOLDER = "https://drive.google.com/drive/folders/1ZPUA8ZUcVKuJWJGo69OaD6W1al8IbDyi?hl=da"
    
    print("="*60)
    print("Download Parquet Files from Google Drive")
    print("="*60)
    
    # Example 1: Download single file
    print("\n### Example 1: Download single file ###\n")
    df = download_parquet_as_dataframe(PARENT_FOLDER, max_depth=5, num_files=1)
    
    if df is not None:
        print("\n" + "="*60)
        print("DataFrame Preview")
        print("="*60)
        print(f"\nFirst 5 rows:")
        print(df.head())
    
    # Example 2: Download 5 files and combine
    print("\n\n" + "="*60)
    print("### Example 2: Download 5 files and combine ###")
    print("="*60)
    df_combined = download_parquet_as_dataframe(PARENT_FOLDER, max_depth=5, num_files=5, combine=True)
    
    if df_combined is not None:
        print("\n" + "="*60)
        print("Combined DataFrame Info")
        print("="*60)
        print(f"\nShape: {df_combined.shape}")
        print(f"Columns: {list(df_combined.columns)}")
        print(f"\nFirst 3 rows:")
        print(df_combined.head(3))
        print(f"\nLast 3 rows:")
        print(df_combined.tail(3))
        
        # Optional: Save to local file
        # output_file = "maritime_data_combined.parquet"
        # df_combined.to_parquet(output_file)
        # print(f"\n✓ Saved to: {output_file}")
    else:
        print("\n❌ Failed to download parquet files")
