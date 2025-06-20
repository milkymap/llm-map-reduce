"""Tests for src.algorithms.prompts module."""

import pytest

from src.algorithms.prompts import map_system_prompt, reduce_system_prompt


class TestPrompts:
    """Test cases for the system prompts."""

    def test_map_system_prompt_exists(self):
        """Test that map_system_prompt is defined and not empty."""
        assert map_system_prompt is not None
        assert isinstance(map_system_prompt, str)
        assert len(map_system_prompt.strip()) > 0

    def test_reduce_system_prompt_exists(self):
        """Test that reduce_system_prompt is defined and not empty."""
        assert reduce_system_prompt is not None
        assert isinstance(reduce_system_prompt, str)
        assert len(reduce_system_prompt.strip()) > 0

    def test_map_system_prompt_content(self):
        """Test that map_system_prompt contains expected content."""
        prompt = map_system_prompt.lower()
        
        # Check for key concepts that should be in the map prompt
        assert 'page' in prompt
        assert 'extract' in prompt or 'key information' in prompt
        assert 'query' in prompt
        assert 'relevant' in prompt

    def test_reduce_system_prompt_content(self):
        """Test that reduce_system_prompt contains expected content."""
        prompt = reduce_system_prompt.lower()
        
        # Check for key concepts that should be in the reduce prompt
        assert 'combine' in prompt or 'merge' in prompt
        assert 'segments' in prompt
        assert 'redundanc' in prompt  # matches 'redundancy' or 'redundancies'
        assert 'query' in prompt

    def test_prompts_are_strings(self):
        """Test that both prompts are string types."""
        assert isinstance(map_system_prompt, str)
        assert isinstance(reduce_system_prompt, str)

    def test_prompts_are_not_identical(self):
        """Test that the two prompts are different."""
        assert map_system_prompt != reduce_system_prompt

    def test_prompts_contain_instructions(self):
        """Test that prompts contain instructional language."""
        # Map prompt should contain instructional language
        map_lower = map_system_prompt.lower()
        assert any(word in map_lower for word in ['you are', 'your task', 'focus on', 'provide'])
        
        # Reduce prompt should contain instructional language
        reduce_lower = reduce_system_prompt.lower()
        assert any(word in reduce_lower for word in ['you are', 'your task', 'focus on', 'ensure'])

    def test_prompts_reasonable_length(self):
        """Test that prompts are of reasonable length (not too short or too long)."""
        # Prompts should be substantial but not excessively long
        assert 50 <= len(map_system_prompt.strip()) <= 2000
        assert 50 <= len(reduce_system_prompt.strip()) <= 2000

    def test_prompts_formatting(self):
        """Test that prompts are properly formatted."""
        # Should not be just whitespace
        assert map_system_prompt.strip() != ""
        assert reduce_system_prompt.strip() != ""
        
        # Should contain actual content (not just newlines and spaces)
        assert len(map_system_prompt.strip()) > 10
        assert len(reduce_system_prompt.strip()) > 10