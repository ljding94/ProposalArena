from setup_llm import get_llm_response
import os
import concurrent.futures
import threading
import csv
import itertools
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def llm_compare_proposal(proposal_folder, IPTS_list, output_folder, prompt_file, llm_model):
    """
    Compare pairs of proposals using LLM based on the provided prompt.
    Saves raw LLM outputs to output_folder/llm_model/comparison and statistics to output_folder/llm_model/comparison_statistics.csv.

    Args:
        proposal_folder (str): Path to the folder containing the .md proposal files
        IPTS_list (list): List of IPTS numbers to compare
        output_folder (str): Base path to save LLM outputs and CSV
        prompt_file (str): Path to the prompt file containing system and user prompts
        llm_model (str): LLM model to use for comparisons

    Returns:
        str: Path to the LLM model folder containing raw outputs and CSV
    """
    print(f"Comparing {len(IPTS_list)} proposals in pairs")

    # Load the prompt template
    with open(prompt_file, "r") as f:
        content = f.read()

    # Split into system and user prompts
    parts = content.split("# user prompt")
    system_prompt = parts[0].replace("# system prompt", "").strip()
    user_template = parts[1].strip()

    print(f"Loaded prompt template: {os.path.basename(prompt_file)}")

    # Sort IPTSs for consistency
    IPTSs = sorted(IPTS_list)
    print(f"Comparing proposals: {IPTSs}")

    # Create llm_model subfolder under output_folder
    llm_model_folder = os.path.join(output_folder, llm_model.split("/")[-1])
    os.makedirs(llm_model_folder, exist_ok=True)

    # Create comparisons folder under llm_model_folder
    comparisons_folder = os.path.join(llm_model_folder, "comparison")
    os.makedirs(comparisons_folder, exist_ok=True)

    print(f"Saving LLM outputs and statistics to {llm_model_folder}")

    # Load existing statistics if CSV exists
    csv_path = os.path.join(llm_model_folder, "comparison_statistics.csv")
    existing_stats = {}
    csv_exists = os.path.exists(csv_path)
    if csv_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = int(row["IPTS_a"])
                b = int(row["IPTS_b"])
                key = (min(a, b), max(a, b))
                # Normalize the row to have IPTS_a < IPTS_b
                row["IPTS_a"] = min(a, b)
                row["IPTS_b"] = max(a, b)
                existing_stats[key] = row
        print(f"Loaded existing statistics from {csv_path} with {len(existing_stats)} pairs")
    else:
        print(f"No existing comparison statistics found, starting fresh")

    # Determine fieldnames: include all columns from existing stats, or default
    if existing_stats:
        all_fieldnames = set()
        for row in existing_stats.values():
            all_fieldnames.update(row.keys())
        fieldnames = sorted(all_fieldnames)
    else:
        fieldnames = ["IPTS_a", "IPTS_b", "input_tokens", "output_tokens", "total_tokens"]

    # Write current existing stats to CSV to ensure it's created with all columns
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_stats.values():
            writer.writerow(row)

    # Lock for thread-safe operations
    lock = threading.Lock()

    total_pairs = len(list(itertools.combinations(IPTSs, 2)))
    done_pairs = set(existing_stats.keys())
    remaining_pairs = [(a, b) for a, b in itertools.combinations(IPTSs, 2) if (min(a, b), max(a, b)) not in done_pairs]
    print(f"Processing {total_pairs} total pairs, {len(remaining_pairs)} remaining (skipping {len(done_pairs)} already done)")

    def process_pair(IPTS_a, IPTS_b):
        try:
            # Ensure IPTS_a < IPTS_b for consistency
            if IPTS_a > IPTS_b:
                IPTS_a, IPTS_b = IPTS_b, IPTS_a

            # Load proposals
            md_path_a = os.path.join(proposal_folder, f"{IPTS_a}.md")
            md_path_b = os.path.join(proposal_folder, f"{IPTS_b}.md")

            with open(md_path_a, "r") as f:
                proposal_text_a = f.read()
            with open(md_path_b, "r") as f:
                proposal_text_b = f.read()

            # Fill the user prompt
            user_content = (
                user_template.replace("{proposal_IPTS_a}", str(IPTS_a))
                .replace("{proposal_IPTS_b}", str(IPTS_b))
                .replace("{proposal_text_a}", proposal_text_a)
                .replace("{proposal_text_b}", proposal_text_b)
            )

            # Get LLM response
            output, usage = get_llm_response(user_content, system_prompt, llm_model)

            # Save raw LLM output
            comparison_file = os.path.join(comparisons_folder, f"{IPTS_a}_vs_{IPTS_b}_llm_comparison.txt")
            with lock:
                with open(comparison_file, "w") as f:
                    f.write(output)

            # Update statistics
            row = {"IPTS_a": IPTS_a, "IPTS_b": IPTS_b, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"]}
            with lock:
                existing_stats[(IPTS_a, IPTS_b)] = row
                # Append the new row to CSV immediately
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)

            return f"Compared {IPTS_a} vs {IPTS_b}, saved output and stats"

        except Exception as e:
            return f"Error comparing {IPTS_a} vs {IPTS_b}: {e}"

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pair, IPTS_a, IPTS_b) for IPTS_a, IPTS_b in remaining_pairs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print(f"All LLM outputs and statistics saved to {llm_model_folder}")
    return llm_model_folder


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
            if val.lower() == "a":
                winner = "A"
            elif val.lower() == "b":
                winner = "B"
            else:
                winner = "tie"
            parsed_ok = True
            fallback_used = True
        else:
            # Alternative loose pattern without quotes/equals
            winner_match2 = re.search(r"\"winner\"\s*:\s*(\"?[AB]|\"?Tie\"?)", output, re.IGNORECASE)
            if winner_match2:
                val = winner_match2.group(1).strip('"')
                if val.lower() == "a":
                    winner = "A"
                elif val.lower() == "b":
                    winner = "B"
                else:
                    winner = "tie"
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


def process_llm_comparison_output(output_folder, llm_model):
    """
    Process saved LLM comparison outputs and update the comparison_statistics.csv with parsed results.

    Args:
        output_folder (str): Base path where LLM model folder is located
        llm_model (str): LLM model used for comparisons

    Returns:
        str: Path to the updated CSV file
    """
    llm_model_folder = os.path.join(output_folder, llm_model.split("/")[-1])
    comparisons_folder = os.path.join(llm_model_folder, "comparison")
    csv_path = os.path.join(llm_model_folder, "comparison_statistics.csv")
    print(f"Processing LLM outputs from {comparisons_folder} and updating {csv_path}")

    # Load existing statistics
    existing_stats = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = int(row["IPTS_a"])
                b = int(row["IPTS_b"])
                key = (min(a, b), max(a, b))
                # Normalize the row
                row["IPTS_a"] = min(a, b)
                row["IPTS_b"] = max(a, b)
                existing_stats[key] = row

    # Find all LLM output files in comparisons_folder
    all_txt_files = [f for f in os.listdir(comparisons_folder) if f.endswith(".txt")]
    print(f"Found {len(all_txt_files)} LLM output files")
    failed_count = 0

    for llm_file in all_txt_files:
        try:
            # Extract IPTS numbers from filename
            m = re.search(r"(\d+)_vs_(\d+)_llm_comparison\.txt$", llm_file)
            if not m:
                raise ValueError(f"Unexpected filename format: {llm_file}")
            IPTS_a = int(m.group(1))
            IPTS_b = int(m.group(2))
            key = (min(IPTS_a, IPTS_b), max(IPTS_a, IPTS_b))

            # Read LLM output
            with open(os.path.join(comparisons_folder, llm_file), "r") as f:
                output = f.read()

            # Parse output
            parsed = parse_comparison_output(output)
            if not parsed.get("parsed_ok", False):
                failed_count += 1

            # Update existing stats
            if key in existing_stats:
                existing_stats[key]["results"] = parsed["results"]
                existing_stats[key]["winner"] = parsed["winner"]
                existing_stats[key]["parsed_ok"] = parsed["parsed_ok"]
            else:
                # If not in stats, perhaps add, but should be there
                print(f"Warning: No stats found for {key}")

            print(f"Processed {IPTS_a} vs {IPTS_b}: winner {parsed['winner']}")

        except Exception as e:
            print(f"Error processing {llm_file}: {e}")

    # Write updated CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["IPTS_a", "IPTS_b", "input_tokens", "output_tokens", "total_tokens", "results", "winner", "parsed_ok"])
        writer.writeheader()
        for row in existing_stats.values():
            writer.writerow(row)

    print(f"Updated comparisons saved to {csv_path}")
    print(f"Parse failures: {failed_count} / {len(all_txt_files)}")
    return csv_path


def calculate_proposal_Bradley_Terry_score(comparison_csv, max_iter=10000, tol=1e-8, tie_mode="half", output_path=None):
    """
    Calculate Bradley-Terry scores (order-independent) from pairwise comparisons.

    The Bradley-Terry model assumes P(i beats j) = s_i / (s_i + s_j), with s_i > 0.
    We estimate s_i via an MM/Ford iterative scheme:
        s_i^{new} = w_i / sum_j n_ij / (s_i + s_j)
    and renormalize to break scale invariance. Ties, if present, are handled as half-wins by default.

    Args:
        comparison_csv (str): Path to the comparison statistics CSV with columns IPTS_a, IPTS_b, results (1=A wins, -1=B wins, 0=tie)
        max_iter (int): Maximum iterations for MM algorithm
        tol (float): Convergence tolerance on relative change
        tie_mode (str): 'half' to count ties as 0.5 win to each; 'ignore' to skip ties
        output_path (str, optional): Explicit path to write BT scores CSV. If None, infer filename next to comparison_csv.

    Returns:
        dict: {IPTS: bt_score} with scores normalized to sum=1
    """
    # Read comparisons
    comparisons = []
    teams = set()
    with open(comparison_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["IPTS_a"])
            b = int(row["IPTS_b"])
            r = int(row["results"])
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
    w = [[0.0] * n for _ in range(n)]  # w[i][j] = times i beat j (can be 0.5 for ties)
    nmat = [[0.0] * n for _ in range(n)]  # nmat[i][j] = matches between i and j

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
            if tie_mode == "half":
                w[i][j] += 0.5
                w[j][i] += 0.5
                nmat[i][j] += 1.0
                nmat[j][i] += 1.0
            else:  # 'ignore'
                continue

    # Total wins per item
    w_i = [sum(w[i][j] for j in range(n) if j != i) for i in range(n)]

    # Initialize skills
    s = [1.0] * n
    eps = 1e-12

    for it in range(max_iter):
        s_new = [0.0] * n
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
        s_new = [max(eps, x / total) for x in s_new]

        # Check convergence (relative change)
        max_rel = max(abs(s_new[i] - s[i]) / (s[i] + eps) for i in range(n))
        s = s_new
        if max_rel < tol:
            break

    scores = {items[i]: s[i] for i in range(n)}

    # Determine output path
    if output_path:
        bt_csv_path = output_path
        os.makedirs(os.path.dirname(bt_csv_path), exist_ok=True)
    else:
        folder = os.path.dirname(comparison_csv)
        bt_csv_path = os.path.join(folder, f"comparison_bt_scores.csv")

    # Extract run cycle from the path
    path_parts = bt_csv_path.split('/')
    run_cycle_dir = [p for p in path_parts if 'SNS_' in p][0]  # e.g., SNS_2014_B
    run_cycle = run_cycle_dir.replace('_', ' ', 1).replace('_', '-')

    # Write CSV
    with open(bt_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["IPTS", "bt_score", "run_cycle"])
        writer.writeheader()
        for ipts, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({"IPTS": ipts, "bt_score": score, "run_cycle": run_cycle})

    print(f"Bradley-Terry scores calculated and saved to {bt_csv_path}")
    return scores, bt_csv_path



def plot_win_lost_heatmap(comparison_csv, bt_csv_path):
    """
    Create a win-loss heatmap from comparison CSV, with axes based on rank from BT score.
    Save the plot to the same folder as bt_csv_path.
    """
    # Read BT scores
    bt_df = pd.read_csv(bt_csv_path)
    # Sort by bt_score descending
    bt_df = bt_df.sort_values('bt_score', ascending=False).reset_index(drop=True)
    proposals = bt_df['IPTS'].tolist()
    N = len(proposals)
    prop_to_idx = {p: i for i, p in enumerate(proposals)}

    # Read comparisons
    comp_df = pd.read_csv(comparison_csv)

    # Initialize matrix
    matrix = np.full((N, N), np.nan)

    # Fill matrix
    for _, row in comp_df.iterrows():
        a = row['IPTS_a']
        b = row['IPTS_b']
        res = row['results']  # 1=A wins, -1=B wins, 0=tie
        if res == 1:
            matrix[prop_to_idx[a]][prop_to_idx[b]] = 1
            matrix[prop_to_idx[b]][prop_to_idx[a]] = -1
        elif res == -1:
            matrix[prop_to_idx[a]][prop_to_idx[b]] = -1
            matrix[prop_to_idx[b]][prop_to_idx[a]] = 1
        else:
            matrix[prop_to_idx[a]][prop_to_idx[b]] = 0
            matrix[prop_to_idx[b]][prop_to_idx[a]] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # Define colors: tomato for loss (-1), gray for tie (0), royalblue for win (1)
    cmap = plt.cm.colors.ListedColormap(['tomato', 'gray', 'royalblue'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    x = np.arange(N + 1)
    y = np.arange(N + 1)
    im = ax.pcolormesh(x, y, matrix, cmap=cmap, norm=norm)

    # Set ticks
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_xticklabels([str(p) for p in proposals], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_yticklabels([str(p) for p in proposals], fontsize=8)
    ax.set_xlabel('IPTS (ranked by BT score descending)', fontsize=10)
    ax.set_ylabel('IPTS (ranked by BT score descending)', fontsize=10)
    ax.set_title('Win-Loss Heatmap', fontsize=12)

    # Save
    folder = os.path.dirname(bt_csv_path)
    plt.savefig(os.path.join(folder, 'win_lost_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Win-loss heatmap saved to {os.path.join(folder, 'win_lost_heatmap.png')}")
