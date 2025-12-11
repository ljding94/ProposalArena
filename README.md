# ProposalArena

ProposalArena is a framework designed to evaluate, compare, and rank research proposals using Large Language Models (LLMs). Specifically tailored for neutron science beamlines (e.g., BL-11A, BL-5, BL-6), this project aims to assess the alignment between LLM-generated evaluations and historical human reviewer rankings, as well as outcome metrics like publication counts.

## Features

*   **Proposal Preprocessing**: Tools to convert and standardize proposal documents (PDF/Markdown) for analysis.
*   **Pairwise Comparison & Bradley-Terry Scoring**: Implements head-to-head proposal comparisons to derive global rankings using the Bradley-Terry model.
*   **Similarity Analysis**: Analyzes and visualizes the similarity between different proposals using embeddings.
*   **Data Visualization**: Generates comprehensive plots for costs, publication statistics, rankings, and similarity matrices.
*   **Parallel Processing**: Efficiently handles large batches of proposals using parallel LLM queries.

## Project Structure

The repository tracks the following core components (data and intermediate outputs are excluded via `.gitignore`):

*   **`script/`**: Core Python source code for the project.
    *   `main.py`: Main entry point for running analyses and orchestrating workflows.
    *   `llm_rank.py`: Logic for pairwise comparisons and ranking proposals.
    *   `llm_score.py`: Logic for individual proposal scoring.
    *   `pre_process.py`: Utilities for converting PDFs to Markdown and cleaning metadata.
    *   `analysis.py`: General analysis and statistical functions.
    *   `requirements.txt`: Python dependency list.
*   **`prompt/`**: Markdown templates for LLM prompts.
    *   Includes templates for plain comparisons, rubric-based scoring, and scientific evaluation.
*   **`doc/`**: Project documentation.
    *   Contains usage guides (e.g., `USAGE_COMPARE_PROPOSALS.md`), outlines, and progress reports.
*   **`plot/`**: Visualization scripts.
    *   Scripts for generating plots related to costs, rankings, similarity, and publication statistics.

> **Note**: The `data/` directory (containing proposals, CSVs, and LLM outputs) and `presentation/` directory are not tracked in the repository and must be set up locally.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd ProposalArena
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r script/requirements.txt
    ```

## Usage

The core logic is orchestrated through `script/main.py`. The script uses command-line arguments to select the operation mode.

### Modes

*   `preprocess_pdf`: Convert proposal PDFs to Markdown.
*   `preprocess_table`: Clean and enrich proposal metadata tables.
*   `llm_ranking`: Run pairwise comparisons and calculate Bradley-Terry scores.
*   `llm_human_analysis`: Compare LLM rankings with human scores and publication metrics.
*   `llm_similarity_analysis`: Analyze proposal similarities using embeddings.

### Running the Analysis

To run a specific mode, use the `--mode` argument:

```bash
# Example: Run LLM ranking
python script/main.py --mode llm_ranking

# Example: Preprocess PDFs
python script/main.py --mode preprocess_pdf
```

Configuration for specific instruments (e.g., EQ-SANS, CNCS, POWGEN) is handled within `script/main.py`.

## Methodology

The project employs a pairwise comparison approach for evaluation:

1.  **Pairwise Comparisons**: LLMs compare proposals in pairs to determine which is better based on specific criteria (e.g., scientific merit).
2.  **Bradley-Terry Model**: The results of these pairwise comparisons are aggregated using the Bradley-Terry model to estimate a score for each proposal, allowing for a global ranking.

## License

[Insert License Information Here]

