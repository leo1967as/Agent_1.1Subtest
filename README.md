# AI Lawyer - Legal Case Database

## Project Overview
AI Lawyer is an advanced legal case database and analysis tool that contains a comprehensive collection of legal decisions and case summaries, with a primary focus on Thai law. The project includes functionality for extracting, storing, and analyzing legal cases, as well as providing a chatbot interface for querying the database.

## Installation
To set up the AI Lawyer project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   ```bash
   cp .env.example .env
   ```

4. Run the application:
   ```bash
   python chatbot_app.py
   ```

## Data Structure
The legal case data is stored in markdown files within the `db_messymd` directory. Each file contains detailed information about specific legal cases, including:

- Case number
- Parties involved
- Legal issues
- Court decisions
- Jurisdiction details

## Key Components

### Case Processing Pipeline
The project includes a series of Python scripts that process legal cases:

- `qctext.py`: Processes raw text from legal documents
- `vector_creation.py`: Creates vector embeddings for case analysis
- `embedder_bge_m3.py`: Implements embedding models for legal text

### Vector Database
The project uses a vector database to store and retrieve case information efficiently. Embeddings are created using advanced techniques to enable semantic search and analysis of legal cases.

## Chatbot Interface
The `chatbot_app.py` script provides a user-friendly interface for querying the legal case database. Users can:

- Search for specific cases
- Analyze legal trends
- Get summaries of court decisions
- Receive legal information in natural language

## Contributing
We welcome contributions to improve the AI Lawyer project. To contribute:

1. Fork the repository and create your branch.
2. Make your changes and commit them.
3. Push to the branch and create a pull request.

Please follow the existing code style and add unit tests for any new features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.