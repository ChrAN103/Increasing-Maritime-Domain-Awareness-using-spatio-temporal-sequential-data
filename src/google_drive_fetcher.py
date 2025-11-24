import re
import os
import requests
import pandas as pd
import io
from typing import List, Dict, Optional, Union


class GoogleDriveFetcher:


    def __init__(self, session: Optional[requests.Session] = None, baseurl_folder: str = "https://drive.google.com/drive/folders"):
        # Reuse a session if provided (e.g. for auth/cookies), else make a new one
        self.session = session or requests.Session() 

        self.baseurl_folder = baseurl_folder
        self.inital_folder = baseurl_folder + "/1oWDXaQPOcnVF8I_bFHrVIYTspjE-eBD_?hl=da"
        self.all_folders = self.list_directory(self.inital_folder)
        self.folder_names = [folder['name'] for folder in self.all_folders]
        self.folder_dict = None


    def _extract_file_id(self, url: str) -> str:
        """
        Extract the Google Drive file ID from common URL formats.
        Raises ValueError if it can't find an ID.
        """
        # /file/d/<id>/
        m = re.search(r"/file/d/([^/]+)/", url)
        if m:
            return m.group(1)

        # /folders/<id>
        m = re.search(r"/folders/([^/?]+)", url)
        if m:
            return m.group(1)

        # ?id=<id>
        m = re.search(r"[?&]id=([^&]+)", url)
        if m:
            return m.group(1)

        # /uc?id=<id>
        m = re.search(r"/uc\?id=([^&]+)", url)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract file ID from URL: {url}")

    def list_directory(self, folder_url: str) -> List[Dict[str, str]]:
        """
        List files and folders in a Google Drive directory.
        
        Args:
            folder_url: Google Drive folder URL
            
        Returns:
            List of dicts with keys: 'id', 'name', 'type' ('file' or 'folder'), 'url'
            
        Note: This is a basic implementation using web scraping. For production,
        use the Google Drive API with OAuth2 authentication.
        """
        folder_id = self._extract_file_id(folder_url)
        
        # Try to get folder listing via the web interface
        # Note: This requires the folder to be publicly accessible
        list_url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        try:
            resp = self.session.get(list_url)
            resp.raise_for_status()
            
            # Parse the HTML to extract file/folder information
            # This is a simplified approach - Google's HTML structure may change
            content = resp.text
            
            items = []
            
            # Extract file IDs and names using regex patterns
            # Pattern for folders: data-id="<id>" ... data-target="doc"
            folder_pattern = r'data-id="([^"]+)"[^>]*data-tooltip="([^"]+)"[^>]*data-type="[^"]*folder'
            for match in re.finditer(folder_pattern, content):
                file_id = match.group(1)
                name = match.group(2)
                items.append({
                    'id': file_id,
                    'name': name,
                    'type': 'folder',
                    'url': f'https://drive.google.com/drive/folders/{file_id}'
                })
            
            # Pattern for files
            file_pattern = r'data-id="([^"]+)"[^>]*data-tooltip="([^"]+)"[^>]*(?!data-type="[^"]*folder)'
            for match in re.finditer(file_pattern, content):
                file_id = match.group(1)
                name = match.group(2)
                # Skip if already added as folder
                if not any(item['id'] == file_id for item in items):
                    items.append({
                        'id': file_id,
                        'name': name,
                        'type': 'file',
                        'url': f'https://drive.google.com/file/d/{file_id}/view'
                    })
            
            return items
            
        except Exception as e:
            # If web scraping fails, provide helpful error message
            raise RuntimeError(
                f"Could not list directory. Error: {e}\n"
                f"Make sure the folder is publicly accessible or use Google Drive API with OAuth2."
            )

    def list_directory_api(self, folder_id: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List files and folders using Google Drive API v3.
        
        Args:
            folder_id: Google Drive folder ID
            api_key: Optional API key for public folders
            
        Returns:
            List of dicts with keys: 'id', 'name', 'type', 'mimeType'
            
        Note: Requires API key or OAuth2 token. For OAuth2, pass authenticated session.
        """
        base_url = "https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents and trashed=false",
            'fields': 'files(id, name, mimeType)',
            'pageSize': 1000
        }
        
        if api_key:
            params['key'] = api_key
        
        try:
            resp = self.session.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            items = []
            for file in data.get('files', []):
                file_type = 'folder' if file['mimeType'] == 'application/vnd.google-apps.folder' else 'file'
                items.append({
                    'id': file['id'],
                    'name': file['name'],
                    'type': file_type,
                    'mimeType': file['mimeType']
                })
            
            return items
            
        except Exception as e:
            raise RuntimeError(f"API request failed: {e}")

    def navigate_to_folder(self, parent_folder_url: str, folder_name: str) -> Optional[str]:
        """
        Find a subfolder by name and return its URL.
        
        Args:
            parent_folder_url: URL of the parent folder
            folder_name: Name of the subfolder to find
            
        Returns:
            URL of the subfolder, or None if not found
        """
        items = self.list_directory(parent_folder_url)
        
        for item in items:
            if item['type'] == 'folder' and item['name'] == folder_name:
                return item['url']
        
        return None

    def download_file_from_folder(self, folder_url: str, file_name: str, dst_path: str) -> Optional[str]:
        """
        Download a specific file from a folder by name.
        
        Args:
            folder_url: URL of the folder containing the file
            file_name: Name of the file to download
            dst_path: Destination path to save the file
            
        Returns:
            Path where file was saved, or None if file not found
        """
        items = self.list_directory(folder_url)
        
        for item in items:
            if item['type'] == 'file' and item['name'] == file_name:
                return self.fetch_to_file(item['url'], dst_path)
        
        return None

    def fetch_bytes(self, url: str) -> bytes:
        """
        Download the file and return its content as bytes.
        """
        file_id = self._extract_file_id(url)
        download_url = "https://drive.google.com/uc?export=download&id=" + file_id

        resp = self.session.get(download_url, stream=True)
        resp.raise_for_status()

        # Handle potential "virus scan" / large file confirmation pages
        # Google sometimes sets a confirmation cookie we need to follow.
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                confirm_url = download_url + f"&confirm={value}"
                resp = self.session.get(confirm_url, stream=True)
                resp.raise_for_status()
                break

        content = resp.content
        return content

    def fetch_to_file(self, url: str, dst_path: str) -> str:
        """
        Download the file and save it to dst_path.
        Returns the path.
        """
        content = self.fetch_bytes(url)

        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(content)

        return dst_path

    def fetch_parquet_to_dataframe(self, url: str) -> pd.DataFrame:
        """
        Download a parquet file from Google Drive and convert it to a pandas DataFrame.
        
        Args:
            url: Google Drive file URL for the parquet file
            
        Returns:
            pandas DataFrame containing the parquet data
            
        Raises:
            ValueError: If the file is not a valid parquet file
        """
        try:
            # Fetch the file content as bytes
            content = self.fetch_bytes(url)
            
            # Convert bytes to DataFrame using pandas
            df = pd.read_parquet(io.BytesIO(content))
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to read parquet file: {e}")

    def download_parquet_from_folder(
        self, 
        folder_url: str, 
        file_name: str, 
        return_dataframe: bool = False
    ) -> Union[Optional[str], Optional[pd.DataFrame]]:
        """
        Download a parquet file from a folder, optionally converting to DataFrame.
        
        Args:
            folder_url: URL of the folder containing the parquet file
            file_name: Name of the parquet file to download
            return_dataframe: If True, return DataFrame instead of saving to disk
            
        Returns:
            If return_dataframe=True: pandas DataFrame or None if file not found
            If return_dataframe=False: path to saved file or None if file not found
        """
        items = self.list_directory(folder_url)
        
        for item in items:
            if item['type'] == 'file' and item['name'] == file_name:
                if return_dataframe:
                    return self.fetch_parquet_to_dataframe(item['url'])
                else:
                    # Default save location
                    dst_path = os.path.join("./downloads", file_name)
                    return self.fetch_to_file(item['url'], dst_path)
        
        return None
    
    def download_folder(self, foldername: str) -> List[pd.DataFrame]: 
        """
        Download all parquet files from a folder and its subfolders recursively.
        
        Args:
            foldername: Name of the folder to download from (must exist in parent directory)
            
        Returns:
            List of DataFrames containing the downloaded parquet files
        """
        if not self.folder_exist(foldername): 
            print(f"{foldername} did not exist")
            return None
        
        # Get the folder URL from the parent directory
        folder_url = self.get_folder_url(foldername)
        
        # Get all items in this folder
        items = self.list_directory(folder_url)
        
        # Recursively download files
        files = self._recursive_download_from_items(items)
        return files 
    
    def _recursive_download_from_items(self, items: List[Dict]) -> List[pd.DataFrame]:
        """
        Recursively download parquet files from a list of items.
        
        Args:
            items: List of item dictionaries from list_directory()
            
        Returns:
            List of DataFrames
        """
        files = []
        
        for item in items:
            # Check if this is a parquet file (has 'Binary' in name)
            if item['type'] == 'file' and 'Binary' in item['name'] and '.parquet' in item['name']:
                try:
                    df = self.fetch_parquet_to_dataframe(item['url'])
                    files.append(df)
                    print(f"Downloaded: {item['name']} ({len(df)} rows)")
                except Exception as e:
                    print(f"Failed to download {item['name']}: {e}")
            
            # If it's a folder (not a file), recursively download from it
            elif 'Shared folder' in item['name']:
                try:
                    sub_items = self.list_directory(item['url'])
                    sub_files = self._recursive_download_from_items(sub_items)
                    files.extend(sub_files)
                except Exception as e:
                    print(f"Failed to access folder {item['name']}: {e}")
        
        return files 
    
    def get_folder_url(self, foldername: str) -> str:
        """
        Get the proper Google Drive URL for a folder by its name.
        Only searches in the initial parent folder (self.all_folders).
        """
        # Find the folder item in all_folders that matches the name
        for folder in self.all_folders:
            if folder['name'] == foldername:
                # Return the URL directly from the folder item
                return folder['url']
        
        # If not found, raise an error
        raise ValueError(f"Folder '{foldername}' not found in parent directory")
    
    def folder_exist(self, foldername: str) -> bool: 
        #make it such that it only appears in the name 
        exists = False 
        for name in self.folder_names:
            if foldername in name: 
                exists = True
        return exists

 




            
            

                


    
        

    


if __name__ == "__main__":
    # Example usage
    gdf = GoogleDriveFetcher()
    
    # Example 1: List directory contents
    folder_url = "https://drive.google.com/drive/folders/1PKdRw01LZYVkb0a2ok--oBuudSFepyQU?hl=da"
    try:
        print("Listing directory contents...")
        items = gdf.list_directory(folder_url)
        print(f"\nFound {len(items)} items:")
        for item in items:
            print(f"  [{item['type']}] {item['name']} (ID: {item['id']})")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Example 2: Navigate to a subfolder
    try:
        print("\n\nNavigating to subfolder 'data'...")
        subfolder_url = gdf.navigate_to_folder(folder_url, "data")
        if subfolder_url:
            print(f"Found subfolder: {subfolder_url}")
        else:
            print("Subfolder 'data' not found")
    except Exception as e:
        print(f"Error navigating: {e}")
    
    # Example 3: Download a specific file from folder
    try:
        print("\n\nDownloading file 'example.csv' from folder...")
        result = gdf.download_file_from_folder(folder_url, "example.csv", "./downloads/example.csv")
        if result:
            print(f"File downloaded to: {result}")
        else:
            print("File 'example.csv' not found in folder")
    except Exception as e:
        print(f"Error downloading: {e}")