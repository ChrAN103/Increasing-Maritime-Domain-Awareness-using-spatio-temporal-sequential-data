"""
Integration test for GoogleDriveFetcher - actually downloads from Google Drive.
This test does NOT use mocks - it performs real network requests.
"""
import unittest
import pandas as pd
from src.google_drive_fetcher import GoogleDriveFetcher


class TestGoogleDriveIntegration(unittest.TestCase):
    """Integration tests that actually connect to Google Drive."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = GoogleDriveFetcher()
        
        # Test URLs - these should be actual accessible Google Drive folders
        self.parent_folder = "https://drive.google.com/drive/folders/1ZPUA8ZUcVKuJWJGo69OaD6W1al8IbDyi?hl=da"
    
    def test_list_directory_real(self):
        """Test listing a real Google Drive directory."""
        print("\n" + "="*60)
        print("TEST: Listing real Google Drive directory")
        print("="*60)
        
        items = self.fetcher.list_directory(self.parent_folder)
        
        print(f"Found {len(items)} items")
        for item in items[:5]:  # Print first 5 items
            print(f"  [{item['type']}] {item['name']}")
            # Debug: print full item
            print(f"      Full item: {item}")
        
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0, "Should find at least one item")
        
        # Check structure
        if items:
            self.assertIn('name', items[0])
            self.assertIn('type', items[0])
            self.assertIn('url', items[0])
    
    def _recursive_find_parquet(self, folder_url, max_depth=5, current_depth=0, max_items_to_check=3):
        """Recursively search for parquet files, handling 'Shared folder' items."""
        if current_depth >= max_depth:
            return None
        
        indent = "  " * current_depth
        try:
            contents = self.fetcher.list_directory(folder_url)
            print(f"{indent}Level {current_depth}: Found {len(contents)} items")
            
            # Debug: show what items look like at deeper levels
            if current_depth >= 3 and contents:
                print(f"{indent}Items at this level:")
                for item in contents[:3]:
                    print(f"{indent}  [{item['type']}] {item['name']}")
            
            # First, look for actual parquet files (exclude items with "Shared folder", but keep "Binary")
            for item in contents:
                if item['type'] == 'file' and '.parquet' in item['name'] and 'Shared folder' not in item['name']:
                    print(f"{indent}✓ Found parquet: {item['name']}")
                    return item
            
            # If no parquet files, recursively check folders (limit to first few)
            folders = [item for item in contents if item['type'] == 'folder' or 'Shared folder' in item['name']]
            print(f"{indent}Checking {min(len(folders), max_items_to_check)} of {len(folders)} folders...")
            
            for item in folders[:max_items_to_check]:  # Only check first few to save time
                print(f"{indent}📁 Checking: {item['name']}")
                
                # Convert to folder URL if needed
                item_url = item['url']
                if '/file/d/' in item_url:
                    item_url = f"https://drive.google.com/drive/folders/{item['id']}"
                
                result = self._recursive_find_parquet(item_url, max_depth, current_depth + 1, max_items_to_check)
                if result:
                    return result
            
            return None
        except Exception as e:
            print(f"{indent}❌ Error: {e}")
            return None
    
    def test_find_and_download_parquet_file(self):
        """Test finding and downloading an actual parquet file."""
        print("\n" + "="*60)
        print("TEST: Finding and downloading real parquet file")
        print("="*60)
        
        print("Recursively searching for parquet files...\n")
        parquet_item = self._recursive_find_parquet(self.parent_folder)
        
        if parquet_item:
            print(f"\n✓ Found file: {parquet_item['name']}")
            print(f"Downloading...")
            
            df = self.fetcher.fetch_parquet_to_dataframe(parquet_item['url'])
            
            print(f"\n✓ Successfully downloaded and converted parquet file!")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            # Assertions
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0, "DataFrame should have rows")
            self.assertGreater(len(df.columns), 0, "DataFrame should have columns")
        else:
            self.fail("No parquet file found in the directory structure")
    
    def _find_first_parquet_recursive(self, folder_url, max_depth=3, current_depth=0):
        """
        Helper function to recursively find and download first parquet file.
        
        Args:
            folder_url: Google Drive folder URL
            max_depth: Maximum recursion depth
            current_depth: Current depth level
            
        Returns:
            pandas DataFrame or None
        """
        if current_depth >= max_depth:
            print("  " * current_depth + "⚠ Max depth reached")
            return None
        
        indent = "  " * current_depth
        print(f"{indent}🔍 Searching at depth {current_depth}...")
        
        try:
            items = self.fetcher.list_directory(folder_url)
            print(f"{indent}Found {len(items)} items")
            
            # First pass: look for parquet files
            for item in items:
                if item['type'] == 'file' and item['name'].endswith('.parquet'):
                    print(f"{indent}✓ Found parquet file: {item['name']}")
                    print(f"{indent}Downloading...")
                    df = self.fetcher.fetch_parquet_to_dataframe(item['url'])
                    print(f"{indent}✓ Successfully converted to DataFrame")
                    return df
            
            # Second pass: search subdirectories
            folders = [item for item in items if item['type'] == 'folder']
            print(f"{indent}No parquet files at this level. Checking {len(folders)} subfolder(s)...")
            
            for folder in folders:
                print(f"{indent}📁 Entering: {folder['name']}")
                result = self._find_first_parquet_recursive(
                    folder['url'], 
                    max_depth, 
                    current_depth + 1
                )
                if result is not None:
                    return result
            
            return None
            
        except Exception as e:
            print(f"{indent}❌ Error: {e}")
            return None
    
    def test_navigate_to_folder(self):
        """Test navigating to a subfolder by name."""
        print("\n" + "="*60)
        print("TEST: Navigate to subfolder")
        print("="*60)
        
        items = self.fetcher.list_directory(self.parent_folder)
        folders = [item for item in items if item['type'] == 'folder']
        
        if folders:
            folder = folders[0]
            print(f"Navigating to folder: {folder['name']}")
            
            folder_url = self.fetcher.navigate_to_folder(self.parent_folder, folder['name'])
            
            self.assertIsNotNone(folder_url, f"Should find folder '{folder['name']}'")
            print(f"✓ Successfully navigated to: {folder['name']}")
            
            # Try to list contents
            contents = self.fetcher.list_directory(folder_url)
            print(f"Folder contains {len(contents)} items")
            self.assertIsInstance(contents, list)
        else:
            print("⚠ No folders found, skipping test")
    
    def test_download_multiple_parquet_files(self):
        """Test downloading multiple parquet files."""
        print("\n" + "="*60)
        print("TEST: Download 5 parquet files")
        print("="*60)
        
        num_files = 5
        print(f"Searching for {num_files} parquet files...\n")
        
        files = []
        self._find_multiple_parquet_recursive(self.parent_folder, max_files=num_files, found_files=files)
        
        print(f"\nFound {len(files)} parquet file(s)")
        self.assertGreaterEqual(len(files), 1, "Should find at least 1 parquet file")
        
        # Download all found files
        dataframes = []
        for i, file_info in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Downloading: {file_info['name']}")
            df = self.fetcher.fetch_parquet_to_dataframe(file_info['url'])
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0, f"DataFrame {i} should have rows")
            
            print(f"  ✓ Shape: {df.shape}")
            dataframes.append(df)
        
        # Test combining DataFrames
        if len(dataframes) > 1:
            print(f"\nCombining {len(dataframes)} DataFrames...")
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            print(f"✓ Combined shape: {combined_df.shape}")
            
            # Verify combined DataFrame
            self.assertIsInstance(combined_df, pd.DataFrame)
            expected_rows = sum(len(df) for df in dataframes)
            self.assertEqual(len(combined_df), expected_rows, "Combined DataFrame should have sum of all rows")
            
            print(f"✓ Total rows: {len(combined_df)}")
            print(f"Columns: {list(combined_df.columns)}")
    
    def _find_multiple_parquet_recursive(self, folder_url, max_depth=5, max_files=5, current_depth=0, found_files=None):
        """Helper to find multiple parquet files."""
        if found_files is None:
            found_files = []
        
        if len(found_files) >= max_files or current_depth >= max_depth:
            return
        
        indent = "  " * current_depth
        print(f"{indent}Level {current_depth}...")
        
        try:
            contents = self.fetcher.list_directory(folder_url)
            
            # Look for parquet files
            for item in contents:
                if item['type'] == 'file' and '.parquet' in item['name'] and 'Shared folder' not in item['name']:
                    if len(found_files) < max_files:
                        print(f"{indent}  ✓ Found: {item['name']}")
                        found_files.append(item)
                        if len(found_files) >= max_files:
                            return
            
            # Check subfolders
            if len(found_files) < max_files:
                folders = [item for item in contents if item['type'] == 'folder' or 'Shared folder' in item['name']]
                
                for folder in folders:
                    if len(found_files) >= max_files:
                        break
                    
                    folder_url_converted = folder['url']
                    if '/file/d/' in folder_url_converted:
                        folder_url_converted = f"https://drive.google.com/drive/folders/{folder['id']}"
                    
                    self._find_multiple_parquet_recursive(folder_url_converted, max_depth, max_files, current_depth + 1, found_files)
        
        except Exception as e:
            print(f"{indent}Error: {e}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
