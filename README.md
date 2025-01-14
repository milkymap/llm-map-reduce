# PDF Map-Reduce with LLM

A powerful tool that leverages Large Language Models (LLM) to process and analyze large PDF documents using the Map-Reduce pattern. This tool enables efficient processing of large documents by breaking them into manageable chunks, processing them in parallel, and then combining the results.

## Features

- Process large PDF documents using Map-Reduce pattern
- Parallel processing with ThreadPoolExecutor
- Interactive query interface
- Configurable context size for processing
- Support for different OpenAI models
- Efficient memory management
- Error handling and graceful interruption

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/milkymap/llm-map-reduce.git]
cd [llm-map-reduce]
```

2. Install the required dependencies:
```bash
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

The tool can be run from the command line with various options:

```bash
python -m src map-reduce -p /path/to/your/file.pdf -m gpt-4o-mini -s 4 -l 64 
```

### Command Line Options

- `-p, --path2file`: Path to the PDF file (required)
- `-m, --model`: Choice of model (default: 'gpt-4o-mini')
  - Options: 'gpt-4o-mini', 'gpt-4o'
- `-s, --context_size`: Number of pages for the map phase (default: 4)

### Interactive Queries

Once running, the tool enters an interactive mode where you can input queries about the document. Press Ctrl+C to exit.

Example:
```bash
query: Summarize the main points of the document
[Summary will appear here]

query: What are the key findings?
[Findings will appear here]
```

## Architecture

The system implements a Map-Reduce pattern:

1. **Map Phase**: 
   - Divides the document into chunks based on context_size
   - Processes each chunk in parallel using the LLM
   - Extracts relevant information based on the query

2. **Reduce Phase**:
   - Combines processed chunks
   - Eliminates redundancies
   - Creates a coherent final response

## Current Limitations

- Requires plain text content in PDF
- Maximum token limit based on OpenAI model constraints

## Error Handling

The system handles several types of errors:
- Empty PDF files
- API failures
- Invalid queries
- Keyboard interruptions

## Future Improvements

Potential areas for enhancement:
- Add streaming responses
- Implement better memory management for very large PDFs
- Add support for more document formats
- Enhance error handling with specific exception types

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- OpenAI for their API
- Click library for CLI interface
- PyPDF2 for PDF processing
- We acknowledge the original LLM×MapReduce paper authors for their inspiring work on long-sequence processing using Large Language Models, which served as a foundation for our implementation.
