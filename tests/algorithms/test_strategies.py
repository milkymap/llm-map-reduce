"""Tests for src.algorithms.strategies module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from src.algorithms.strategies import llm_map, llm_reduce, llm_map_reduce


class TestLLMMap:
    """Test cases for the llm_map function."""

    def test_llm_map_success(self, mock_openai_client, mock_chat_completion, sample_query):
        """Test successful llm_map execution."""
        # Setup
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        page_content = "Sample page content"
        model = "gpt-4o-mini"
        
        # Execute
        result = llm_map(page_content, sample_query, model, mock_openai_client)
        
        # Assert
        assert result == "Test response content"
        mock_openai_client.chat.completions.create.assert_called_once()
        
        # Check the call arguments
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['model'] == model
        assert call_args[1]['max_tokens'] == 1024
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        assert sample_query in call_args[1]['messages'][1]['content']
        assert page_content in call_args[1]['messages'][1]['content']

    def test_llm_map_with_empty_page(self, mock_openai_client, mock_chat_completion, sample_query):
        """Test llm_map with empty page content."""
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        
        result = llm_map("", sample_query, "gpt-4o-mini", mock_openai_client)
        
        assert result == "Test response content"
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch('src.algorithms.strategies.logger')
    def test_llm_map_logs_info(self, mock_logger, mock_openai_client, mock_chat_completion, sample_query):
        """Test that llm_map logs the map phase."""
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        
        llm_map("content", sample_query, "gpt-4o-mini", mock_openai_client)
        
        mock_logger.info.assert_called_with('map phase')


class TestLLMReduce:
    """Test cases for the llm_reduce function."""

    def test_llm_reduce_success(self, mock_openai_client, mock_chat_completion, sample_query):
        """Test successful llm_reduce execution."""
        # Setup
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        accumulator = ["First segment", "Second segment", "Third segment"]
        model = "gpt-4o-mini"
        
        # Execute
        result = llm_reduce(accumulator, model, mock_openai_client, sample_query)
        
        # Assert
        assert result == "Test response content"
        mock_openai_client.chat.completions.create.assert_called_once()
        
        # Check the call arguments
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['model'] == model
        assert call_args[1]['max_tokens'] == 1024
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        assert sample_query in call_args[1]['messages'][1]['content']

    def test_llm_reduce_with_empty_accumulator(self, mock_openai_client, sample_query):
        """Test llm_reduce with empty accumulator."""
        result = llm_reduce([], "gpt-4o-mini", mock_openai_client, sample_query)
        
        assert result is None
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_llm_reduce_filters_none_values(self, mock_openai_client, mock_chat_completion, sample_query):
        """Test llm_reduce filters out None values from accumulator."""
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        accumulator = ["First segment", None, "Second segment", None]
        
        result = llm_reduce(accumulator, "gpt-4o-mini", mock_openai_client, sample_query)
        
        assert result == "Test response content"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_llm_reduce_with_all_none_values(self, mock_openai_client, sample_query):
        """Test llm_reduce with all None values in accumulator."""
        accumulator = [None, None, None]
        
        result = llm_reduce(accumulator, "gpt-4o-mini", mock_openai_client, sample_query)
        
        assert result is None
        mock_openai_client.chat.completions.create.assert_not_called()

    @patch('src.algorithms.strategies.logger')
    def test_llm_reduce_logs_info(self, mock_logger, mock_openai_client, mock_chat_completion, sample_query):
        """Test that llm_reduce logs the reduce phase."""
        mock_openai_client.chat.completions.create.return_value = mock_chat_completion
        
        llm_reduce(["segment"], "gpt-4o-mini", mock_openai_client, sample_query)
        
        mock_logger.info.assert_called_with('reduce phase')


class TestLLMMapReduce:
    """Test cases for the llm_map_reduce function."""

    def test_llm_map_reduce_empty_pages(self, mock_openai_client, sample_query):
        """Test llm_map_reduce with empty pages."""
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, 4, [])
        
        assert result is None
        mock_openai_client.chat.completions.create.assert_not_called()

    @patch('src.algorithms.strategies.llm_map')
    def test_llm_map_reduce_small_pages_direct_map(self, mock_llm_map, mock_openai_client, sample_query):
        """Test llm_map_reduce with pages less than context_size calls llm_map directly."""
        mock_llm_map.return_value = "Direct map result"
        pages = ["page1", "page2"]
        context_size = 4
        
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
        
        assert result == "Direct map result"
        mock_llm_map.assert_called_once_with(
            page="page1\npage2",
            model="gpt-4o-mini",
            llm=mock_openai_client,
            query=sample_query
        )

    @patch('src.algorithms.strategies.llm_reduce')
    @patch('src.algorithms.strategies.ThreadPoolExecutor')
    def test_llm_map_reduce_large_pages_uses_threads(self, mock_executor_class, mock_llm_reduce, mock_openai_client, sample_query):
        """Test llm_map_reduce with large pages uses ThreadPoolExecutor."""
        # Setup
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_executor.map.return_value = ["result1", "result2", "result3"]
        mock_llm_reduce.return_value = "Final reduced result"
        
        pages = [f"page{i}" for i in range(12)]  # 12 pages
        context_size = 4
        
        # Execute
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
        
        # Assert
        assert result == "Final reduced result"
        mock_executor_class.assert_called_once_with(max_workers=4)  # 12 pages with batch_size=3: [0:3], [3:6], [6:9], [9:12] = 4 partitions
        mock_executor.map.assert_called_once()
        mock_llm_reduce.assert_called_once()

    def test_llm_map_reduce_partition_calculation(self, mock_openai_client, sample_query):
        """Test that llm_map_reduce correctly calculates partitions."""
        with patch('src.algorithms.strategies.ThreadPoolExecutor') as mock_executor_class, \
             patch('src.algorithms.strategies.llm_reduce') as mock_llm_reduce:
            
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            mock_executor.map.return_value = ["result"]
            mock_llm_reduce.return_value = "Final result"
            
            pages = [f"page{i}" for i in range(8)]  # 8 pages
            context_size = 4
            
            llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
            
            # With 8 pages and context_size 4: batch_size = 8/4 = 2
            # Partitions: [0:2], [2:4], [4:6], [6:8] = 4 partitions
            mock_executor_class.assert_called_once_with(max_workers=4)

    @patch('src.algorithms.strategies.llm_map')
    def test_llm_map_reduce_single_page_equal_context_size(self, mock_llm_map, mock_openai_client, sample_query):
        """Test llm_map_reduce when pages equal context_size."""
        mock_llm_map.return_value = "Single result"
        pages = ["page1", "page2", "page3", "page4"]
        context_size = 4  # Same as number of pages
        
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
        
        assert result == "Single result"
        mock_llm_map.assert_called_once()

    @patch('src.algorithms.strategies.llm_map')
    @patch('src.algorithms.strategies.llm_reduce')
    def test_llm_map_reduce_integration_with_partitioning(self, mock_llm_reduce, mock_llm_map, mock_openai_client, sample_query):
        """Test llm_map_reduce with partitioning logic."""
        # Mock the map function to return simple responses
        mock_llm_map.return_value = "mapped_content"
        mock_llm_reduce.return_value = "Final result"
        
        pages = [f"page{i}" for i in range(6)]  # 6 pages
        context_size = 2  # batch_size = 3, creates 2 partitions
        
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
        
        # Should eventually call reduce
        assert result == "Final result"
        mock_llm_reduce.assert_called()


class TestLLMMapReduceIntegration:
    """Integration tests for the map-reduce workflow."""

    @patch('src.algorithms.strategies.llm_map')
    @patch('src.algorithms.strategies.llm_reduce')
    def test_full_map_reduce_workflow(self, mock_llm_reduce, mock_llm_map, mock_openai_client, sample_query):
        """Test the complete map-reduce workflow."""
        # Setup
        mock_llm_map.return_value = "Mapped content"
        mock_llm_reduce.return_value = "Final reduced content"
        
        pages = [f"Content of page {i+1}" for i in range(6)]
        context_size = 2
        
        # Execute
        result = llm_map_reduce(sample_query, "gpt-4o-mini", mock_openai_client, context_size, pages)
        
        # Assert
        assert result == "Final reduced content"
        # Should eventually call reduce
        mock_llm_reduce.assert_called()

    def test_map_reduce_preserves_parameters(self, mock_openai_client, sample_query):
        """Test that parameters are correctly passed through the map-reduce chain."""
        with patch('src.algorithms.strategies.llm_map') as mock_llm_map, \
             patch('src.algorithms.strategies.llm_reduce') as mock_llm_reduce:
            
            mock_llm_map.return_value = "Mapped result"
            mock_llm_reduce.return_value = "Final result"
            
            pages = ["page1", "page2"]
            model = "gpt-4o"
            context_size = 1
            
            llm_map_reduce(sample_query, model, mock_openai_client, context_size, pages)
            
            # Check that parameters are passed correctly
            if mock_llm_map.called:
                call_args = mock_llm_map.call_args
                assert call_args[1]['query'] == sample_query
                assert call_args[1]['model'] == model
                assert call_args[1]['llm'] == mock_openai_client
            
            if mock_llm_reduce.called:
                call_args = mock_llm_reduce.call_args
                assert call_args[1]['query'] == sample_query
                assert call_args[1]['model'] == model
                assert call_args[1]['llm'] == mock_openai_client