import unittest
import os
import tempfile
import shutil
import pandas as pd
import io
from unittest.mock import Mock, patch, MagicMock
from src.google_drive_fetcher import GoogleDriveFetcher


class TestGoogleDriveFetcher(unittest.TestCase):
    """Unit tests for GoogleDriveFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = GoogleDriveFetcher()
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample folder URL for testing
        self.folder_url = "https://drive.google.com/drive/folders/1PKdRw01LZYVkb0a2ok--oBuudSFepyQU"
        self.file_url = "https://drive.google.com/file/d/1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl/view"

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that GoogleDriveFetcher initializes correctly."""
        self.assertIsNotNone(self.fetcher)
        self.assertIsNotNone(self.fetcher.session)

    def test_initialization_with_custom_session(self):
        """Test initialization with custom session."""
        import requests
        custom_session = requests.Session()
        fetcher = GoogleDriveFetcher(session=custom_session)
        self.assertEqual(fetcher.session, custom_session)

    def test_extract_file_id_from_file_url(self):
        """Test extracting file ID from /file/d/ URL format."""
        url = "https://drive.google.com/file/d/1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl/view"
        file_id = self.fetcher._extract_file_id(url)
        self.assertEqual(file_id, "1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl")

    def test_extract_file_id_from_folder_url(self):
        """Test extracting file ID from /folders/ URL format."""
        url = "https://drive.google.com/drive/folders/1PKdRw01LZYVkb0a2ok--oBuudSFepyQU"
        file_id = self.fetcher._extract_file_id(url)
        self.assertEqual(file_id, "1PKdRw01LZYVkb0a2ok--oBuudSFepyQU")

    def test_extract_file_id_from_folder_url_with_params(self):
        """Test extracting file ID from folder URL with query parameters."""
        url = "https://drive.google.com/drive/folders/1PKdRw01LZYVkb0a2ok--oBuudSFepyQU?hl=da"
        file_id = self.fetcher._extract_file_id(url)
        self.assertEqual(file_id, "1PKdRw01LZYVkb0a2ok--oBuudSFepyQU")

    def test_extract_file_id_from_open_url(self):
        """Test extracting file ID from ?id= URL format."""
        url = "https://drive.google.com/open?id=1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl"
        file_id = self.fetcher._extract_file_id(url)
        self.assertEqual(file_id, "1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl")

    def test_extract_file_id_from_uc_url(self):
        """Test extracting file ID from /uc?id= URL format."""
        url = "https://drive.google.com/uc?id=1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl&export=download"
        file_id = self.fetcher._extract_file_id(url)
        self.assertEqual(file_id, "1WvMMXTLLlQ2MWgkCytob2SwF7SPTdsCl")

    def test_extract_file_id_invalid_url(self):
        """Test that invalid URL raises ValueError."""
        url = "https://example.com/invalid/url"
        with self.assertRaises(ValueError) as context:
            self.fetcher._extract_file_id(url)
        self.assertIn("Could not extract file ID", str(context.exception))

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_returns_list(self, mock_get):
        """Test that list_directory returns a list of items."""
        # Mock HTML response with parquet files and folders
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="file123" data-tooltip="data.parquet" data-type="file"></div>
        <div data-id="folder456" data-tooltip="parquet" data-type="folder"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        items = self.fetcher.list_directory(self.folder_url)
        
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_parquet_files(self, mock_get):
        """Test listing directory with parquet files."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="parquet1" data-tooltip="file1.parquet" data-type="file"></div>
        <div data-id="parquet2" data-tooltip="file2.parquet" data-type="file"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        items = self.fetcher.list_directory(self.folder_url)
        
        # Check for parquet files
        parquet_files = [item for item in items if item['name'].endswith('.parquet')]
        self.assertGreater(len(parquet_files), 0)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_parquet_folder(self, mock_get):
        """Test listing directory with folder named 'parquet'."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="folder_parquet" data-tooltip="parquet" data-type="folder"></div>
        <div data-id="file1" data-tooltip="data.parquet" data-type="file"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        items = self.fetcher.list_directory(self.folder_url)
        
        # Check for folder named 'parquet'
        parquet_folders = [item for item in items if item['name'] == 'parquet' and item['type'] == 'folder']
        self.assertEqual(len(parquet_folders), 1)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_item_structure(self, mock_get):
        """Test that list_directory returns items with correct structure."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="file123" data-tooltip="test.parquet" data-type="file"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        items = self.fetcher.list_directory(self.folder_url)
        
        if items:
            item = items[0]
            self.assertIn('id', item)
            self.assertIn('name', item)
            self.assertIn('type', item)
            self.assertIn('url', item)
            self.assertIn(item['type'], ['file', 'folder'])

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_error_handling(self, mock_get):
        """Test that list_directory handles errors properly."""
        mock_get.side_effect = Exception("Network error")

        with self.assertRaises(RuntimeError) as context:
            self.fetcher.list_directory(self.folder_url)
        
        self.assertIn("Could not list directory", str(context.exception))

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_navigate_to_folder_success(self, mock_get):
        """Test successfully navigating to a subfolder named 'parquet'."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="folder_parquet" data-tooltip="parquet" data-type="folder"></div>
        <div data-id="other_folder" data-tooltip="other" data-type="folder"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        folder_url = self.fetcher.navigate_to_folder(self.folder_url, "parquet")
        
        self.assertIsNotNone(folder_url)
        self.assertIn("folder_parquet", folder_url)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_navigate_to_folder_not_found(self, mock_get):
        """Test navigating to non-existent folder."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="folder1" data-tooltip="other" data-type="folder"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        folder_url = self.fetcher.navigate_to_folder(self.folder_url, "nonexistent")
        
        self.assertIsNone(folder_url)

    @patch('src.google_drive_fetcher.GoogleDriveFetcher.fetch_to_file')
    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_download_file_from_folder_parquet(self, mock_get, mock_fetch):
        """Test downloading a parquet file from folder."""
        # Mock directory listing
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="parquet_file" data-tooltip="data.parquet" data-type="file"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock file download
        mock_fetch.return_value = os.path.join(self.temp_dir, "data.parquet")

        result = self.fetcher.download_file_from_folder(
            self.folder_url, 
            "data.parquet", 
            os.path.join(self.temp_dir, "data.parquet")
        )
        
        self.assertIsNotNone(result)
        mock_fetch.assert_called_once()

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_download_file_from_folder_not_found(self, mock_get):
        """Test downloading non-existent file returns None."""
        mock_response = Mock()
        mock_response.text = '''
        <div data-id="file1" data-tooltip="other.txt" data-type="file"></div>
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.fetcher.download_file_from_folder(
            self.folder_url, 
            "nonexistent.parquet", 
            os.path.join(self.temp_dir, "test.parquet")
        )
        
        self.assertIsNone(result)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_fetch_bytes(self, mock_get):
        """Test fetching file as bytes."""
        mock_response = Mock()
        mock_response.content = b"test parquet data"
        mock_response.cookies = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        data = self.fetcher.fetch_bytes(self.file_url)
        
        self.assertIsInstance(data, bytes)
        self.assertEqual(data, b"test parquet data")

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_fetch_bytes_with_virus_warning(self, mock_get):
        """Test fetching file with virus scan warning."""
        # First response with warning cookie
        mock_response1 = Mock()
        mock_response1.content = b"warning page"
        mock_response1.cookies = {'download_warning_123': 'confirm_token'}
        mock_response1.raise_for_status = Mock()
        
        # Second response with actual content
        mock_response2 = Mock()
        mock_response2.content = b"actual parquet data"
        mock_response2.raise_for_status = Mock()
        
        mock_get.side_effect = [mock_response1, mock_response2]

        data = self.fetcher.fetch_bytes(self.file_url)
        
        self.assertEqual(data, b"actual parquet data")
        self.assertEqual(mock_get.call_count, 2)

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_fetch_to_file(self, mock_get):
        """Test downloading file to disk."""
        mock_response = Mock()
        mock_response.content = b"parquet file content"
        mock_response.cookies = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        dst_path = os.path.join(self.temp_dir, "test.parquet")
        result_path = self.fetcher.fetch_to_file(self.file_url, dst_path)
        
        self.assertEqual(result_path, dst_path)
        self.assertTrue(os.path.exists(dst_path))
        
        with open(dst_path, 'rb') as f:
            content = f.read()
        self.assertEqual(content, b"parquet file content")

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_fetch_to_file_creates_directory(self, mock_get):
        """Test that fetch_to_file creates parent directories."""
        mock_response = Mock()
        mock_response.content = b"parquet data"
        mock_response.cookies = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        dst_path = os.path.join(self.temp_dir, "subdir", "nested", "test.parquet")
        result_path = self.fetcher.fetch_to_file(self.file_url, dst_path)
        
        self.assertTrue(os.path.exists(dst_path))
        self.assertTrue(os.path.exists(os.path.dirname(dst_path)))

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_api(self, mock_get):
        """Test listing directory using Google Drive API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'files': [
                {
                    'id': 'file1',
                    'name': 'data1.parquet',
                    'mimeType': 'application/octet-stream'
                },
                {
                    'id': 'folder1',
                    'name': 'parquet',
                    'mimeType': 'application/vnd.google-apps.folder'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        items = self.fetcher.list_directory_api("folder_id_123", api_key="test_key")
        
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]['type'], 'file')
        self.assertEqual(items[1]['type'], 'folder')
        self.assertEqual(items[1]['name'], 'parquet')

    @patch('src.google_drive_fetcher.requests.Session.get')
    def test_list_directory_api_error(self, mock_get):
        """Test API error handling."""
        mock_get.side_effect = Exception("API Error")

        with self.assertRaises(RuntimeError) as context:
            self.fetcher.list_directory_api("folder_id")
        
        self.assertIn("API request failed", str(context.exception))

    def test_multiple_parquet_files_in_folder(self):
        """Test scenario with multiple parquet files in a folder."""
        with patch('src.google_drive_fetcher.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.text = '''
            <div data-id="p1" data-tooltip="file1.parquet" data-type="file"></div>
            <div data-id="p2" data-tooltip="file2.parquet" data-type="file"></div>
            <div data-id="p3" data-tooltip="file3.parquet" data-type="file"></div>
            <div data-id="folder_p" data-tooltip="parquet" data-type="folder"></div>
            '''
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            items = self.fetcher.list_directory(self.folder_url)
            
            parquet_files = [item for item in items if item['name'].endswith('.parquet')]
            parquet_folders = [item for item in items if item['name'] == 'parquet' and item['type'] == 'folder']
            
            self.assertEqual(len(parquet_files), 3)
            self.assertEqual(len(parquet_folders), 1)

    def test_fetch_parquet_to_dataframe_success(self):
        """Test successful conversion of parquet file to DataFrame."""
        # Create mock parquet data
        mock_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        mock_df.to_parquet(buffer)
        mock_parquet_bytes = buffer.getvalue()
        
        with patch.object(self.fetcher, 'fetch_bytes', return_value=mock_parquet_bytes):
            result_df = self.fetcher.fetch_parquet_to_dataframe(self.file_url)
            
            # Verify DataFrame is returned
            self.assertIsInstance(result_df, pd.DataFrame)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(result_df, mock_df)
            self.assertEqual(len(result_df), 3)
            self.assertEqual(list(result_df.columns), ['col1', 'col2', 'col3'])

    def test_fetch_parquet_to_dataframe_with_complex_data(self):
        """Test parquet to DataFrame conversion with various data types."""
        mock_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.1, 2.2, 3.3, 4.4],
            'str_col': ['a', 'b', 'c', 'd'],
            'bool_col': [True, False, True, False],
            'datetime_col': pd.date_range('2023-01-01', periods=4, freq='h')
        })
        
        buffer = io.BytesIO()
        mock_df.to_parquet(buffer)
        mock_parquet_bytes = buffer.getvalue()
        
        with patch.object(self.fetcher, 'fetch_bytes', return_value=mock_parquet_bytes):
            result_df = self.fetcher.fetch_parquet_to_dataframe(self.file_url)
            
            pd.testing.assert_frame_equal(result_df, mock_df)
            self.assertEqual(result_df['int_col'].dtype, mock_df['int_col'].dtype)
            self.assertEqual(result_df['float_col'].dtype, mock_df['float_col'].dtype)
            self.assertEqual(result_df['bool_col'].dtype, mock_df['bool_col'].dtype)

    def test_fetch_parquet_to_dataframe_empty_dataframe(self):
        """Test conversion of empty parquet file to DataFrame."""
        mock_df = pd.DataFrame()
        
        buffer = io.BytesIO()
        mock_df.to_parquet(buffer)
        mock_parquet_bytes = buffer.getvalue()
        
        with patch.object(self.fetcher, 'fetch_bytes', return_value=mock_parquet_bytes):
            result_df = self.fetcher.fetch_parquet_to_dataframe(self.file_url)
            
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), 0)

    def test_fetch_parquet_to_dataframe_invalid_data(self):
        """Test that invalid parquet data raises ValueError."""
        # Mock invalid parquet data
        invalid_data = b"This is not a valid parquet file"
        
        with patch.object(self.fetcher, 'fetch_bytes', return_value=invalid_data):
            with self.assertRaises(ValueError) as context:
                self.fetcher.fetch_parquet_to_dataframe(self.file_url)
            
            self.assertIn("Failed to read parquet file", str(context.exception))

    def test_fetch_parquet_to_dataframe_fetch_bytes_error(self):
        """Test that fetch_bytes errors are propagated correctly."""
        with patch.object(self.fetcher, 'fetch_bytes', side_effect=Exception("Network error")):
            with self.assertRaises(ValueError) as context:
                self.fetcher.fetch_parquet_to_dataframe(self.file_url)
            
            self.assertIn("Failed to read parquet file", str(context.exception))

    def test_download_parquet_from_folder_to_dataframe(self):
        """Test downloading and converting parquet file to DataFrame from folder."""
        # Mock directory listing
        mock_items = [
            {'name': 'data.parquet', 'type': 'file', 'url': 'https://drive.google.com/file/d/123/view'},
            {'name': 'other.txt', 'type': 'file', 'url': 'https://drive.google.com/file/d/456/view'}
        ]
        
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['x', 'y', 'z']
        })
        
        with patch.object(self.fetcher, 'list_directory', return_value=mock_items), \
             patch.object(self.fetcher, 'fetch_parquet_to_dataframe', return_value=mock_df):
            
            result = self.fetcher.download_parquet_from_folder(
                self.folder_url, 
                'data.parquet', 
                return_dataframe=True
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, mock_df)

    def test_download_parquet_from_folder_to_file(self):
        """Test downloading parquet file to disk from folder."""
        mock_items = [
            {'name': 'data.parquet', 'type': 'file', 'url': 'https://drive.google.com/file/d/123/view'}
        ]
        
        expected_path = "./downloads/data.parquet"
        
        with patch.object(self.fetcher, 'list_directory', return_value=mock_items), \
             patch.object(self.fetcher, 'fetch_to_file', return_value=expected_path):
            
            result = self.fetcher.download_parquet_from_folder(
                self.folder_url, 
                'data.parquet', 
                return_dataframe=False
            )
            
            self.assertEqual(result, expected_path)
            self.fetcher.fetch_to_file.assert_called_once_with(
                'https://drive.google.com/file/d/123/view',
                expected_path
            )

    def test_download_parquet_from_folder_file_not_found(self):
        """Test that None is returned when file is not found in folder."""
        mock_items = [
            {'name': 'other.parquet', 'type': 'file', 'url': 'https://drive.google.com/file/d/123/view'}
        ]
        
        with patch.object(self.fetcher, 'list_directory', return_value=mock_items):
            result_df = self.fetcher.download_parquet_from_folder(
                self.folder_url, 
                'missing.parquet', 
                return_dataframe=True
            )
            
            self.assertIsNone(result_df)
            
            result_path = self.fetcher.download_parquet_from_folder(
                self.folder_url, 
                'missing.parquet', 
                return_dataframe=False
            )
            
            self.assertIsNone(result_path)

    def test_download_parquet_from_folder_ignores_folders(self):
        """Test that folders are ignored when searching for parquet files."""
        mock_items = [
            {'name': 'data.parquet', 'type': 'folder', 'url': 'https://drive.google.com/drive/folders/123'},
            {'name': 'data.parquet', 'type': 'file', 'url': 'https://drive.google.com/file/d/456/view'}
        ]
        
        mock_df = pd.DataFrame({'col': [1, 2]})
        
        with patch.object(self.fetcher, 'list_directory', return_value=mock_items), \
             patch.object(self.fetcher, 'fetch_parquet_to_dataframe', return_value=mock_df):
            
            result = self.fetcher.download_parquet_from_folder(
                self.folder_url, 
                'data.parquet', 
                return_dataframe=True
            )
            
            # Should use the file, not the folder
            self.fetcher.fetch_parquet_to_dataframe.assert_called_once_with(
                'https://drive.google.com/file/d/456/view'
            )


if __name__ == '__main__':
    unittest.main()
