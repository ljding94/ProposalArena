import csv
import json
import os
import itertools
import concurrent.futures
import threading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from setup_llm import get_llm_response
import re
from scipy.stats import spearmanr, kendalltau

def parse_comparison_output(output):
    """
    Parse the LLM output for comparison results.

    Strategy:
    1) Try to parse JSON (handles fenced code blocks and minor escaping).
    2) If JSON parse fails, fallback to regex extraction for winner (A/B/Tie).

    Args:
        output (str): Raw LLM output

    Returns:
        dict: Parsed results containing winner, comparison_table, recommendation, and raw output
    """
    raw = output
    output = (output or "").strip()

    # Unwrap code fences if present
    if output.startswith("```"):
        # Remove the first fence line and the trailing ``` if present
        lines = output.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines).strip()

    # Try JSON parsing first (including extracting a JSON block if extra text surrounds it)
    parsed_ok = False
    comparison_table = ""
    recommendation = ""
    winner = "tie"

    # Attempt to find the largest JSON-like block
    json_candidate = output
    m = re.search(r"\{[\s\S]*\}", output)
    if m:
        json_candidate = m.group(0)

    # Replace backslashes conservatively
    try:
        result_json = json.loads(json_candidate)
        comparison_table = result_json.get("comparison_table", "")
        recommendation = result_json.get("recommendation", "")
        winner = result_json.get("winner", winner)
        parsed_ok = True
    except Exception as e:
        # Fall back to regex-based winner extraction
        print(f"Invalid JSON: {e}")
        fallback_used = False
        # Look for patterns like: "winner": A, "winner": "A", winner: B, Winner = "Tie"
        winner_match = re.search(r"\bwinner\b\s*[:=]\s*\"?('?)(A|B|Tie|TIE|tie)\1\"?", output, re.IGNORECASE)
        if winner_match:
            val = winner_match.group(2)
            if val.lower() == 'a':
                winner = 'A'
            elif val.lower() == 'b':
                winner = 'B'
            else:
                winner = 'tie'
            parsed_ok = True
            fallback_used = True
        else:
            # Alternative loose pattern without quotes/equals
            winner_match2 = re.search(r"\"winner\"\s*:\s*(\"?[AB]|\"?Tie\"?)", output, re.IGNORECASE)
            if winner_match2:
                val = winner_match2.group(1).strip('"')
                if val.lower() == 'a':
                    winner = 'A'
                elif val.lower() == 'b':
                    winner = 'B'
                else:
                    winner = 'tie'
                parsed_ok = True
                fallback_used = True
        if fallback_used:
            print(f"Fallback regex extraction used for winner: {winner}")

    # Determine results value
    res = 0
    if winner == "A":
        res = 1
    elif winner == "B":
        res = -1

    return {
        "winner": winner,
        "results": res,
        "comparison_table": comparison_table,
        "recommendation": recommendation if parsed_ok else "Could not parse LLM output",
        "raw_output": raw,
        "parsed_ok": parsed_ok,
    }


def process_llm_comparison_outputs(llm_data_folder, llm_model, prompt_type="plain"):
    """
    Process saved LLM outputs and generate comparison results.

    Args:
        folder (str): Path to the folder containing the .md files
        llm_data_folder (str): Path to the folder containing saved LLM outputs
        llm_model (str): LLM model used for comparisons
        prompt_type (str): Which comparison prompt template was used: one of {"plain","rubric","science"}
    """

    llm_model_folder = os.path.join(llm_data_folder, llm_model.replace('/', '_'))
    comparisons_folder = os.path.join(llm_model_folder, 'comparisons')
    print(f"Processing LLM outputs from {comparisons_folder}")

    # Initialize CSV with header in llm_model_folder
    # Include prompt_type in output filename for clarity
    csv_path = os.path.join(
        llm_model_folder,
        f"{prompt_type}_proposal_comparisons.csv"
    )
    with open(csv_path, 'w', newline='') as f:
        # Also include llm_model and prompt_type columns for provenance
        writer = csv.DictWriter(f, fieldnames=["IPTS_a", "IPTS_b", "results", "llm_model", "prompt_type"])
        writer.writeheader()

    # Find all LLM output files in comparisons_folder
    # Prefer files that match the current prompt_type prefix; if none, fall back to any .txt for backward compatibility
    expected_prefix = f"{prompt_type}_statement_of_resch"
    all_txt_files = [f for f in os.listdir(comparisons_folder) if f.endswith('.txt')]
    llm_files = [f for f in all_txt_files if f.startswith(expected_prefix)]
    if not llm_files:
        print(f"No files matching prompt_type '{prompt_type}'; falling back to all .txt files (legacy).")
        llm_files = all_txt_files
    print(f"Found {len(llm_files)} LLM output files")
    failed_count = 0

    for llm_file in llm_files:
        # Extract IPTS numbers from filename
        # Expected formats (supported):
        #  - {prompt_type}_statement_of_resch{A}_vs_{B}_llm_comparison.txt (new format)
        #  - statement_of_resch{A}_vs_{B}_llm_comparison.txt (legacy format)
        try:
            # Robust regex to capture A and B (support both new and legacy formats)
            m = re.search(r"(?:.*_)?statement_of_resch(\d+)_vs_(\d+)_llm_comparison\.txt$", llm_file)
            if not m:
                raise ValueError(f"Unexpected filename format: {llm_file}")
            IPTS_a = int(m.group(1))
            IPTS_b = int(m.group(2))

            # Read LLM output
            with open(os.path.join(comparisons_folder, llm_file), 'r') as f:
                output = f.read()

            # Parse output
            parsed = parse_comparison_output(output)
            if not parsed.get("parsed_ok", False):
                failed_count += 1

            # Append result to CSV
            result_row = {
                "IPTS_a": IPTS_a,
                "IPTS_b": IPTS_b,
                "results": parsed['results'],
                "llm_model": llm_model,
                "prompt_type": prompt_type,
            }
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["IPTS_a", "IPTS_b", "results", "llm_model", "prompt_type"])
                writer.writerow(result_row)

            print(f"Processed {IPTS_a} vs {IPTS_b}: winner {parsed['winner']}")

        except Exception as e:
            print(f"Error processing {llm_file}: {e}")

    print(f"All comparisons saved to {csv_path}")
    print(f"Parse failures: {failed_count} / {len(llm_files)}")


def compare_proposals_parallel(
    IPTSs,
    folder,
    llm_data_folder=None,
    llm_model="google/gemini-2.5-flash-preview-09-2025",
    prompt_type="plain",
):
    """
    Compare pairs of proposals in parallel using LLM based on the reviewer compare prompt.
    Saves raw LLM outputs to data/LLM_data folder.

    Args:
        IPTSs (list): List of proposal numbers to compare
        folder (str): Path to the folder containing the .md files
        llm_model (str): LLM model to use for comparisons
        llm_data_folder (str, optional): Path to save LLM outputs. If None, creates timestamped folder
        prompt_type (str): Which comparison prompt template to use: one of {"plain","rubric","science"}

    Returns:
        str: Path to the LLM data folder containing raw outputs
    """
    print(f"Comparing {len(IPTSs)} proposals in parallel")

    # Load the prompt template based on prompt_type
    prompt_map = {
        "empty": "./prompt/prompt_compare_empty.md",
        "plain": "./prompt/prompt_compare_plain.md",
        "rubric": "./prompt/prompt_compare_rubric.md",
        "science": "./prompt/prompt_compare_science.md",
    }
    if prompt_type not in prompt_map:
        print(f"Unknown prompt_type '{prompt_type}', defaulting to 'plain'.")
        prompt_type = "plain"
    prompt_file = prompt_map[prompt_type]
    with open(prompt_file, 'r') as f:
        content = f.read()

    # Split into system and user prompts
    parts = content.split("# user prompt")
    system_prompt = parts[0].replace("# system prompt", "").strip()
    user_template = parts[1].strip()

    print(f"Loaded prompt template: {os.path.basename(prompt_file)}")

    # Sort IPTSs for consistency
    IPTSs = sorted(IPTSs)
    print(f"Comparing proposals: {IPTSs}")
    # Create llm_model subfolder under llm_data_folder
    if llm_data_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_data_folder = os.path.join("data", "LLM_data", f"{timestamp}")
    llm_model_folder = os.path.join(llm_data_folder, llm_model.replace('/', '_'))
    os.makedirs(llm_model_folder, exist_ok=True)
    print(f"Saving LLM outputs to {llm_model_folder}")

    # Create comparisons folder under llm_model_folder
    comparisons_folder = os.path.join(llm_model_folder, 'comparisons')
    os.makedirs(comparisons_folder, exist_ok=True)

    # Lock for thread-safe operations
    lock = threading.Lock()

    total_pairs = len(list(itertools.combinations(IPTSs, 2)))
    print(f"Processing {total_pairs} pairs in parallel")

    def process_pair(IPTS_a, IPTS_b):
        try:
            # Load proposals
            md_path_a = os.path.join(folder, f"statement_of_resch{IPTS_a}.md")
            md_path_b = os.path.join(folder, f"statement_of_resch{IPTS_b}.md")

            with open(md_path_a, 'r') as f:
                proposal_text_a = f.read()
            with open(md_path_b, 'r') as f:
                proposal_text_b = f.read()

            # Fill the user prompt (now includes IPTS placeholders)
            user_content = (
                user_template
                .replace("{proposal_IPTS_a}", str(IPTS_a))
                .replace("{proposal_IPTS_b}", str(IPTS_b))
                .replace("{proposal_text_a}", proposal_text_a)
                .replace("{proposal_text_b}", proposal_text_b)
            )

            # Get LLM response
            output, _ = get_llm_response(user_content, system_prompt, llm_model)

            # Save raw LLM output only to comparisons folder; include model and prompt type in filename
            comparison_file = os.path.join(
                comparisons_folder,
                f"{prompt_type}_statement_of_resch{IPTS_a}_vs_{IPTS_b}_llm_comparison.txt"
            )
            with lock:
                with open(comparison_file, 'w') as f:
                    f.write(output)

            return f"Queried LLM for {IPTS_a} vs {IPTS_b}, saved to {comparison_file}"

        except Exception as e:
            return f"Error comparing {IPTS_a} vs {IPTS_b}: {e}"

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pair, IPTS_a, IPTS_b) for IPTS_a, IPTS_b in itertools.combinations(IPTSs, 2)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print(f"All LLM outputs saved to {llm_model_folder}")
    return llm_model_folder


def calculate_Bradley_Terry_score(csv_path, max_iter=10000, tol=1e-8, tie_mode='half', output_path=None):
    """
    Calculate Bradley-Terry scores (order-independent) from pairwise comparisons.

    The Bradley-Terry model assumes P(i beats j) = s_i / (s_i + s_j), with s_i > 0.
    We estimate s_i via an MM/Ford iterative scheme:
        s_i^{new} = w_i / sum_j n_ij / (s_i + s_j)
    and renormalize to break scale invariance. Ties, if present, are handled as half-wins by default.

    Args:
        csv_path (str): Path to the proposal_comparisons.csv file with columns IPTS_a, IPTS_b, results (1=A wins, -1=B wins, 0=tie)
        max_iter (int): Maximum iterations for MM algorithm
        tol (float): Convergence tolerance on relative change
        tie_mode (str): 'half' to count ties as 0.5 win to each; 'ignore' to skip ties
        output_path (str, optional): Explicit path to write BT scores CSV. If None, infer filename next to csv_path.

    Returns:
        dict: {IPTS: bt_score} with scores normalized to sum=1
    """
    # Read comparisons
    comparisons = []
    teams = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row['IPTS_a'])
            b = int(row['IPTS_b'])
            r = int(row['results'])
            comparisons.append((a, b, r))
            teams.add(a)
            teams.add(b)

    if not comparisons:
        print("No comparisons found for Bradley-Terry.")
        return {}

    # Index mapping
    items = sorted(teams)
    idx = {item: i for i, item in enumerate(items)}
    n = len(items)

    # Initialize matrices
    w = [[0.0]*n for _ in range(n)]  # w[i][j] = times i beat j (can be 0.5 for ties)
    nmat = [[0.0]*n for _ in range(n)]  # nmat[i][j] = matches between i and j

    # Populate from comparisons
    for a, b, r in comparisons:
        i = idx[a]
        j = idx[b]
        if r == 1:
            w[i][j] += 1.0
            nmat[i][j] += 1.0
            nmat[j][i] += 1.0
        elif r == -1:
            w[j][i] += 1.0
            nmat[i][j] += 1.0
            nmat[j][i] += 1.0
        else:  # tie
            if tie_mode == 'half':
                w[i][j] += 0.5
                w[j][i] += 0.5
                nmat[i][j] += 1.0
                nmat[j][i] += 1.0
            else:  # 'ignore'
                continue

    # Total wins per item
    w_i = [sum(w[i][j] for j in range(n) if j != i) for i in range(n)]

    # Initialize skills
    s = [1.0]*n
    eps = 1e-12

    for it in range(max_iter):
        s_new = [0.0]*n
        for i in range(n):
            denom = 0.0
            s_i = s[i]
            for j in range(n):
                if i == j:
                    continue
                nij = nmat[i][j]
                if nij > 0:
                    denom += nij / (s_i + s[j] + eps)
            if denom > 0:
                s_new[i] = w_i[i] / (denom + eps)
            else:
                # No matches for i; keep previous skill
                s_new[i] = s_i

        # Normalize to sum = 1 to fix scale
        total = sum(s_new)
        if total <= 0:
            # Fallback: keep previous
            s_new = s
            total = sum(s_new) + eps
        s_new = [max(eps, x/total) for x in s_new]

        # Check convergence (relative change)
        max_rel = max(abs(s_new[i]-s[i]) / (s[i] + eps) for i in range(n))
        s = s_new
        if max_rel < tol:
            break

    scores = {items[i]: s[i] for i in range(n)}

    # Determine output path
    if output_path:
        bt_csv_path = output_path
        os.makedirs(os.path.dirname(bt_csv_path), exist_ok=True)
    else:
        folder = os.path.dirname(csv_path)
        basename = os.path.basename(csv_path)
        if basename.endswith(".csv"):
            model_part = basename[:-len(".csv")]
            bt_csv_path = os.path.join(folder, f"{model_part}_proposal_bt_scores.csv")
        else:
            bt_csv_path = os.path.join(folder, "bt_scores.csv")

    # Write CSV
    with open(bt_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["IPTS", "bt_score"])
        writer.writeheader()
        for ipts, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({"IPTS": ipts, "bt_score": round(score, 6)})

    print(f"Bradley-Terry scores calculated and saved to {bt_csv_path}")
    return scores, bt_csv_path


