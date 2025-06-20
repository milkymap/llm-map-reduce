"""Tests for src.settings module."""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from src.settings import Credentials


class TestCredentials:
    """Test cases for the Credentials class."""

    def test_credentials_with_valid_api_key(self):
        """Test Credentials initialization with valid OpenAI API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key-123'}):
            credentials = Credentials()
            assert credentials.openai_api_key == 'test-api-key-123'

    def test_credentials_missing_api_key(self):
        """Test Credentials initialization fails without OpenAI API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Credentials()
            
            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert errors[0]['type'] == 'missing'
            # The loc contains the environment variable name since we use validation_alias
            assert 'OPENAI_API_KEY' in errors[0]['loc'] or 'openai_api_key' in errors[0]['loc']

    def test_credentials_empty_api_key(self):
        """Test Credentials initialization accepts empty OpenAI API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            credentials = Credentials()
            assert credentials.openai_api_key == ''

    def test_credentials_validation_alias(self):
        """Test that Credentials uses the correct environment variable name."""
        # Test with correct env var name
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'correct-key'}, clear=True):
            credentials = Credentials()
            assert credentials.openai_api_key == 'correct-key'
        
        # Test with wrong env var name - should fail
        with patch.dict(os.environ, {'WRONG_API_KEY': 'wrong-key'}, clear=True):
            with pytest.raises(ValidationError):
                Credentials()

    def test_credentials_field_properties(self):
        """Test that the openai_api_key field has correct properties."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            credentials = Credentials()
            
            # Check that the field is required
            field_info = credentials.model_fields['openai_api_key']
            assert field_info.is_required()
            
            # Check the validation alias
            assert field_info.validation_alias == 'OPENAI_API_KEY'