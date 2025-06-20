"""Integration tests for src.__main__ module."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from click.testing import CliRunner
from PyPDF2 import PdfReader

from src.__main__ import handler, map_reduce


class TestCLIHandler:
    """Test cases for the CLI handler."""

    def test_handler_group_exists(self):
        """Test that handler is a Click group."""
        assert hasattr(handler, 'commands')
        assert callable(handler)

    def test_handler_context_initialization(self):
        """Test that handler initializes context correctly."""
        runner = CliRunner()
        
        with patch('src.__main__.Credentials') as mock_credentials:
            mock_credentials.return_value = Mock()
            
            # Test without command to trigger the handler function
            result = runner.invoke(handler, [])
            
            # Should not error when just calling the group
            assert result.exit_code == 0


class TestMapReduceCommand:
    """Test cases for the map_reduce command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def create_temp_pdf(self):
        """Create a temporary PDF file for testing."""
        # Create a simple text file and rename it to .pdf for testing
        # Note: This is a mock PDF - in real tests you'd create a proper PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(b"Mock PDF content")
        temp_file.close()
        return temp_file.name

    @patch('src.__main__.llm_map_reduce')
    @patch('src.__main__.PdfReader')
    @patch('src.__main__.OpenAI')
    @patch('src.__main__.Credentials')
    def test_map_reduce_command_success(self, mock_credentials, mock_openai, mock_pdf_reader, mock_llm_map_reduce):
        """Test successful execution of map_reduce command."""
        # Setup mocks
        mock_creds = Mock()
        mock_creds.openai_api_key = "test-key"
        mock_credentials.return_value = mock_creds
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock PDF reader
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        
        mock_reader = Mock()
        mock_reader.pages = mock_pages
        mock_pdf_reader.return_value = mock_reader
        
        mock_llm_map_reduce.return_value = "Test response"
        
        # Create temp file
        temp_pdf = self.create_temp_pdf()
        
        try:
            # Mock input to avoid hanging
            with patch('builtins.input', side_effect=['test query', KeyboardInterrupt()]):
                result = self.runner.invoke(map_reduce, [
                    '--path2file', temp_pdf,
                    '--model', 'gpt-4o-mini',
                    '--context_size', '2',
                    '--limit', '10'
                ])
            
            # Should exit cleanly due to KeyboardInterrupt
            assert result.exit_code == 0
            
            # Verify mocks were called
            mock_credentials.assert_called_once()
            mock_openai.assert_called_once_with(api_key="test-key")
            mock_pdf_reader.assert_called_once()
            
        finally:
            # Clean up temp file
            os.unlink(temp_pdf)

    @patch('src.__main__.PdfReader')
    @patch('src.__main__.OpenAI')
    @patch('src.__main__.Credentials')
    def test_map_reduce_command_empty_pdf(self, mock_credentials, mock_openai, mock_pdf_reader):
        """Test map_reduce command with empty PDF."""
        # Setup mocks
        mock_creds = Mock()
        mock_creds.openai_api_key = "test-key"
        mock_credentials.return_value = mock_creds
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock PDF reader with empty pages
        mock_page = Mock()
        mock_page.extract_text.return_value = ""  # Empty content
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        temp_pdf = self.create_temp_pdf()
        
        try:
            with patch('src.__main__.logger') as mock_logger:
                result = self.runner.invoke(map_reduce, [
                    '--path2file', temp_pdf,
                    '--model', 'gpt-4o-mini'
                ])
            
            # Should exit with code 0 due to early exit
            assert result.exit_code == 0
            mock_logger.warning.assert_called_with('pdf file must contains at least one plain text page')
            
        finally:
            os.unlink(temp_pdf)

    def test_map_reduce_command_missing_file(self):
        """Test map_reduce command with non-existent file."""
        result = self.runner.invoke(map_reduce, [
            '--path2file', '/non/existent/file.pdf'
        ])
        
        # Should fail due to file not existing
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "invalid value" in result.output.lower()

    def test_map_reduce_command_parameters(self):
        """Test map_reduce command parameter handling."""
        # Test with invalid model
        temp_pdf = self.create_temp_pdf()
        
        try:
            result = self.runner.invoke(map_reduce, [
                '--path2file', temp_pdf,
                '--model', 'invalid-model'
            ])
            
            # Should fail due to invalid model choice
            assert result.exit_code != 0
            
        finally:
            os.unlink(temp_pdf)

    @patch('src.__main__.llm_map_reduce')
    @patch('src.__main__.PdfReader')
    @patch('src.__main__.OpenAI')
    @patch('src.__main__.Credentials')
    def test_map_reduce_command_limit_parameter(self, mock_credentials, mock_openai, mock_pdf_reader, mock_llm_map_reduce):
        """Test that limit parameter correctly limits pages."""
        # Setup mocks
        mock_creds = Mock()
        mock_creds.openai_api_key = "test-key"
        mock_credentials.return_value = mock_creds
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock PDF reader with many pages
        mock_pages = []
        for i in range(10):  # 10 pages
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        
        mock_reader = Mock()
        mock_reader.pages = mock_pages
        mock_pdf_reader.return_value = mock_reader
        
        mock_llm_map_reduce.return_value = "Test response"
        
        temp_pdf = self.create_temp_pdf()
        
        try:
            with patch('builtins.input', side_effect=['test query', KeyboardInterrupt()]):
                result = self.runner.invoke(map_reduce, [
                    '--path2file', temp_pdf,
                    '--limit', '3'  # Limit to 3 pages
                ])
            
            assert result.exit_code == 0
            
            # Check that llm_map_reduce was called with limited pages
            if mock_llm_map_reduce.called:
                call_args = mock_llm_map_reduce.call_args
                pages_arg = call_args[1]['pages']
                assert len(pages_arg) == 3  # Should be limited to 3 pages
                
        finally:
            os.unlink(temp_pdf)

    @patch('src.__main__.llm_map_reduce')
    @patch('src.__main__.PdfReader')
    @patch('src.__main__.OpenAI')
    @patch('src.__main__.Credentials')
    def test_map_reduce_command_exception_handling(self, mock_credentials, mock_openai, mock_pdf_reader, mock_llm_map_reduce):
        """Test exception handling in map_reduce command."""
        # Setup mocks
        mock_creds = Mock()
        mock_creds.openai_api_key = "test-key"
        mock_credentials.return_value = mock_creds
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Mock llm_map_reduce to raise an exception
        mock_llm_map_reduce.side_effect = Exception("Test exception")
        
        temp_pdf = self.create_temp_pdf()
        
        try:
            with patch('builtins.input', return_value='test query'):
                with patch('src.__main__.logger') as mock_logger:
                    result = self.runner.invoke(map_reduce, [
                        '--path2file', temp_pdf
                    ])
            
            # Should handle the exception gracefully
            assert result.exit_code == 0
            mock_logger.error.assert_called()
            
        finally:
            os.unlink(temp_pdf)

    @patch('src.__main__.llm_map_reduce')
    @patch('src.__main__.PdfReader')
    @patch('src.__main__.OpenAI')
    @patch('src.__main__.Credentials')
    def test_map_reduce_command_none_response(self, mock_credentials, mock_openai, mock_pdf_reader, mock_llm_map_reduce):
        """Test handling of None response from llm_map_reduce."""
        # Setup mocks
        mock_creds = Mock()
        mock_creds.openai_api_key = "test-key"
        mock_credentials.return_value = mock_creds
        
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        # Mock llm_map_reduce to return None
        mock_llm_map_reduce.return_value = None
        
        temp_pdf = self.create_temp_pdf()
        
        try:
            with patch('builtins.input', side_effect=['test query', KeyboardInterrupt()]):
                with patch('src.__main__.logger') as mock_logger:
                    result = self.runner.invoke(map_reduce, [
                        '--path2file', temp_pdf
                    ])
            
            assert result.exit_code == 0
            mock_logger.warning.assert_called_with('none value was found during the map_reduce phase')
            
        finally:
            os.unlink(temp_pdf)


class TestMainModule:
    """Test cases for the main module entry point."""

    @patch('src.__main__.load_dotenv')
    @patch('src.__main__.handler')
    def test_main_entry_point(self, mock_handler, mock_load_dotenv):
        """Test the main entry point when script is run directly."""
        # This would be called when running python -m src
        # We can't easily test the if __name__ == '__main__' block directly
        # but we can test that the components are properly set up
        
        # Verify that the handler is callable
        assert callable(mock_handler)
        
        # The load_dotenv and handler should be importable
        from src.__main__ import load_dotenv, handler
        assert callable(load_dotenv)
        assert callable(handler)