"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, MagicMock
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock(spec=OpenAI)
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@pytest.fixture
def mock_chat_completion():
    """Create a mock ChatCompletion object."""
    # Create a mock message
    mock_message = Mock(spec=ChatCompletionMessage)
    mock_message.content = "Test response content"
    
    # Create a mock choice
    mock_choice = Mock(spec=Choice)
    mock_choice.message = mock_message
    
    # Create a mock completion
    mock_completion = Mock(spec=ChatCompletion)
    mock_completion.choices = [mock_choice]
    
    return mock_completion


@pytest.fixture
def sample_pages():
    """Sample pages for testing."""
    return [
        "This is the first page of the document. It contains information about topic A.",
        "Second page discusses topic B and provides more details.",
        "Third page covers topic C with additional context.",
        "Fourth page concludes with topic D and summary information."
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What are the main topics discussed in the document?"


@pytest.fixture
def sample_model():
    """Sample model name for testing."""
    return "gpt-4o-mini"


@pytest.fixture
def mock_pdf_reader():
    """Create a mock PDF reader."""
    mock_reader = Mock()
    mock_pages = []
    
    for i in range(4):
        mock_page = Mock()
        mock_page.extract_text.return_value = f"Sample text from page {i+1}"
        mock_pages.append(mock_page)
    
    mock_reader.pages = mock_pages
    return mock_reader


@pytest.fixture
def empty_pages():
    """Empty pages list for testing edge cases."""
    return []


@pytest.fixture
def single_page():
    """Single page for testing."""
    return ["This is a single page document with some content."]


@pytest.fixture
def large_pages_list():
    """Large list of pages for testing pagination."""
    return [f"This is page {i+1} content" for i in range(20)]