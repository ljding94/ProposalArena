### Project Outline: Evaluating LLM-Based Proposal Review Using Rubric Scoring and Pairwise Elo Ranking

This outline provides a structured plan for implementing and testing the two approaches (rubric-based scoring and pairwise comparisons with Elo ranking) to evaluate user proposals at a national lab facility like ORNL Neutron Sciences. The goal is to assess how well LLM-generated rankings align with an "ideal" ranking based on publication counts (or other outcome metrics) from past cycles, compared to human reviewer rankings. We'll use the provided rubric from "SRC Reviewer Guide.pdf" as the core evaluation framework, and the two sample proposals ("statement_of_resch30212.txt" on wormlike micelles and "statement_of_resch30235.txt" on mucus structure) for initial testing. Assume you have a dataset of ~20 proposals per cycle, with associated historical scores and publication data.

The project will be implemented in Python, using an LLM API (e.g., Grok API via x.ai) for evaluations. Start with a pilot on the two samples, then scale to full cycles. Estimated timeline: 1-2 weeks for setup and pilot, plus iteration based on results.

#### 1. Project Setup and Data Preparation
   - **Objective**: Prepare inputs for LLM evaluation, ensuring consistency and anonymity.
   - **Steps**:
     - Extract and format the rubric: Use pages 1-3 and 7-8 from "SRC Reviewer Guide.pdf" as the core prompt content. Focus on "Providing Substantive Review Comments" (key questions on importance, contribution, neutron suitability, publication potential), "Proposal Rating Scale" (5-point scale with descriptors), and "Reviewing Collaborative Development Proposals" if applicable (both samples appear to be General User type, but check proposal headers).
       - Convert to a clean text string for prompts (e.g., via OCR if needed for images, but text is already provided).
       - Key excerpt for prompts: Include the 5-point scale definitions and questions to address (e.g., "What is the importance of their primary scientific question...").
     - Process proposals: Use your existing script to convert all proposals to TXT files. For each:
       - Add metadata: Assign unique IDs (e.g., Proposal A = 30212, Proposal B = 30235) and track publication counts (simulate for pilot: e.g., assume Proposal A has 5 pubs, B has 3; replace with real data later).
     - Dataset organization: Create a directory with TXT files, a CSV for metadata (columns: proposal_id, file_path, human_score_avg, publication_count, ideal_rank).
     - Handle multi-instrument/collaborative types: Check headers; if multi-instrument (e.g., NSE + SANS in samples), score per instrument or average as per rubric (page 7).
     - Tools: Python (pandas for CSV, os/pathlib for file handling).
   - **Pilot Test**: Use the two samples to verify formatting. Output: A prompt-ready rubric string and two anonymized proposal texts.

#### 2. Approach 1: Rubric-Based Scoring
   - **Objective**: Use LLM to score each proposal individually based on the rubric, generate rankings, and compare to ideal/human baselines.
   - **Prompt Design**:
     - Base prompt: "You are an expert reviewer for ORNL Neutron Sciences. Evaluate the following proposal using this rubric: [insert full rubric text]. Provide a score on the 5-point scale (e.g., 4.5), a brief rationale (100-200 words) addressing the key questions, and ensure your score reflects scientific merit, innovation, neutron suitability, and publication potential. Proposal: [insert proposal text]. Output in JSON: {'score': float, 'rationale': str}."
     - Enhancements: Add Chain-of-Thought (CoT) – "First, summarize the proposal's scientific question. Then, evaluate each rubric criterion step-by-step before assigning the score."
     - Few-shot: Include 1-2 anonymized examples from past data (if available) with known scores.
     - Variability control: Set LLM temperature to 0.2 for consistency; run 3-5 times per proposal and average scores.
   - **Scoring Process**:
     - Loop through proposals: For each TXT, feed into LLM via API.
     - Aggregate: Compute average score if multi-runs; handle half-points (e.g., 4.5) as per rubric.
     - Generate ranking: Sort proposals by descending score to get LLM rank.
   - **Implementation**:
     - Script: Python function `score_proposal(proposal_text, rubric_text)` using API calls.
     - Batch: Process all ~20 proposals in parallel if API allows.
     - Output: CSV with proposal_id, llm_score, llm_rationale, llm_rank.
   - **Pilot Test**: Score the two samples. Expected: Proposal 30212 (wormlike micelles under shear – novel, with prelim data) might score ~4.0-4.5; 30235 (mucus structure – applied bio, strong prelim SANS) ~4.0. Rank them and compare (e.g., if pubs: A > B, check alignment).

#### 3. Approach 2: Pairwise Comparisons with Elo Ranking
   - **Objective**: Use LLM for head-to-head comparisons, then apply Elo to derive a global ranking; compare to baselines.
   - **Prompt Design**:
     - Base prompt: "You are an expert reviewer for ORNL Neutron Sciences. Compare Proposal A and Proposal B using this rubric: [insert rubric text]. Decide which is better overall based on scientific merit, innovation, neutron suitability, and publication potential. Output: 'A' if A is better, 'B' if B is better, 'tie' if equal. Provide a brief rationale (50-100 words). Proposal A: [text A]. Proposal B: [text B]. Output in JSON: {'winner': str, 'rationale': str}."
     - Enhancements: CoT – "Evaluate each on the rubric criteria, then compare relative strengths."
     - Variability: Run each pair 3-5 times; majority vote for winner.
   - **Comparison Process**:
     - Generate pairs: For N proposals, create all unique pairs (N*(N-1)/2; ~190 for 20). Use itertools.combinations in Python.
     - Run comparisons: Feed pairs to LLM in batches.
     - Elo computation: Use Python's `elo` library or implement simple Elo (start all at 1500 rating; update based on wins/ties with K-factor=32). Run multiple tournaments if multi-runs.
     - Generate ranking: Sort by final Elo scores.
   - **Implementation**:
     - Script: Functions `compare_pair(text_a, text_b, rubric_text)` and `compute_elo(results_df)`.
     - Output: CSV with pair_id, winner, rationale; plus final elo_ranks.csv.
   - **Pilot Test**: Compare the two samples (one pair). Expected: If 30212 wins (more fundamental/novel), Elo would rank it higher. Scale to simulated 5-10 proposals for full test.

#### 4. Evaluation and Comparison
   - **Objective**: Quantify how well LLM rankings perform vs. human and ideal.
   - **Baselines**:
     - Human ranking: From historical score data (avg per proposal).
     - Ideal ranking: Sort by publication count (normalize by time since approval, citations if available).
   - **Metrics**:
     - Rank correlation: Spearman's Rho or Kendall's Tau between LLM rank, human rank, and ideal rank.
     - Accuracy: % agreement in top-5/bottom-5; mean absolute rank error.
     - Compare approaches: Compute correlations for Approach 1 vs. 2; test if pairwise > scoring.
     - Bias analysis: Check LLM rationales for patterns (e.g., overfavoring bio vs. physics proposals).
   - **Steps**:
     - Compute for pilot: With two samples, simple (e.g., do ranks match ideal?).
     - Full eval: Run on one historical cycle; use bootstrap for CI on correlations.
     - Visualization: Tables/plots of ranks; heatmaps for pairwise wins.
   - **Improvements Based on Results**: If correlation <0.7, refine prompts (e.g., add domain-specific examples); test different LLMs.

#### 5. Tools, Resources, and Best Practices
   - **Tech Stack**: Python 3.12+ (pandas, json, requests for API); LLM API (Grok-4 preferred for reasoning); Jupyter for prototyping.
   - **Handling LLM Issues**: Rate limiting – batch small; cost – estimate ~$0.01-0.05 per eval; consistency – multi-runs + low temp.
   - **Scalability**: For full cycles, use cloud (e.g., AWS) if >100 proposals.
   - **Ethical/Practical**: Ensure no real data leakage; get approval if using sensitive proposals. Document all prompts for reproducibility. (let's use provider: Google Vertex since they don't store data) https://openrouter.ai/docs/features/privacy-and-logging
   - **Next Steps After Outline**: Implement pilot scripts; run on samples; share results for feedback before scaling.

This outline is modular—start with Section 1-2, then pilot Approach 1, then 2. If you provide publication data for the samples or more details, I can help refine prompts or sketch code snippets!