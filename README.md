# Medical Diagnosis Expert System

An advanced, AI-driven Medical Diagnosis System that leverages Natural Language Processing (NLP), Knowledge Graphs, and Bayesian Inference to diagnose diseases based on patient symptoms.

## Features

- **NLP Symptom Extraction**: Uses `spaCy` to process natural language inputs and accurately extract medical symptoms from patient descriptions.
- **Knowledge Graph Integration**: Powered by Neo4j, the system maps the complex relationships between diseases and their respective symptoms.
- **Probabilistic Inference**: Uses `pgmpy` (Discrete Bayesian Networks) to calculate the exact probability of diseases based on observed symptoms.
- **Patient Session Management**: Saves patient data (CNIC, age, location, symptom history) persistently, allowing sessions to be resumed later.
- **Interactive CLI Menu**: Easy-to-use command-line interface for managing patients and diagnosing diseases.

## Technologies Used

- **Python 3.x**
- **Neo4j**: Graph database for storing the medical knowledge graph.
- **spaCy**: For Natural Language Processing and text lemmatization.
- **pgmpy**: For Bayesian Network inference and probability calculations.

## Prerequisites

- Python 3.8+
- [Neo4j Desktop or Neo4j Community Edition](https://neo4j.com/download/) installed and running.
- Neo4j database configured with:
  - **URI**: `bolt://127.0.0.1:7687`
  - **Username**: `neo4j`
  - **Password**: `luna123*` *(Change in code if your local setup differs)*

## Installation

1. Clone this repository:
   ```bash
   git clone <your_github_repo_link_here>
   cd <repository_folder>
   ```

2. Install the required Python packages:
   ```bash
   pip install spacy neo4j pgmpy
   ```

3. Download the `spaCy` English language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Ensure your Neo4j database is running and credentials match the script configuration.
5. Ensure a `knowledge.txt` file exists in the root directory formatted as `Disease has symptoms Symptom1, Symptom2.` to populate the graph on first run.

## Usage

Start the system by running:

```bash
python medical_Diagnosis.py
```

### Workflows

- **Start/Resume Session**: Enter a patient's CNIC (format: XXXXX-XXXXXXX-X). If the patient exists, the system will resume their session; otherwise, it will prompt for basic details to create a new profile.
- **Add Symptoms**: Describe symptoms naturally (e.g., "I have a severe headache and fever"). The NLP engine will parse and extract them.
- **Run Diagnosis**: The system will query the Neo4j Knowledge Graph and use Bayesian inference to list the Top 3 most probable diseases with percentage likelihoods.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
