import csv
import json
import os
import re
import concurrent.futures
import threading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from setup_llm import get_llm_response


def score_proposals(IPTSs, folder, shot=0):
    """
    Score proposals using LLM based on the reviewer prompt.

    Args:
        IPTSs (list): List of proposal numbers
        folder (str): Path to the folder containing the .md files
    """
    print(f"processing {len(IPTSs)} proposals")

    # Load the prompt template
    prompt_file = f"./reviewer_score_prompt_{shot}_shot.md"
    with open(prompt_file, 'r') as f:
        content = f.read()

    # Split into system and user prompts
    parts = content.split("# user prompt")
    system_prompt = parts[0].replace("# system prompt", "").strip()
    user_template = parts[1].strip()

    print("system_prompt:", system_prompt)
    print("user_template:", user_template)

    scores = []
    for i in range(len(IPTSs)):
        IPTS = IPTSs[i]
        md_path = f"{folder}/statement_of_resch{IPTS}.md"
        print(f"processing proposal: {IPTS}, {i}/{len(IPTSs)}")
        try:
            # Read the proposal markdown
            with open(md_path, 'r') as f:
                proposal_text = f.read()

            # Fill the user prompt
            user_content = user_template.replace("{proposal_text}", proposal_text)

            # print("system_prompt:", system_prompt)
            # print("user_content:", user_content)

            # Get LLM response
            output, _ = get_llm_response(user_content, system_prompt)

            # Parse the JSON output
            output = output.strip()
            if output.startswith("```json") and output.endswith("```"):
                output = output[7:-3].strip()
            # Escape backslashes to handle LaTeX in reasoning
            output = output.replace('\\', '\\\\')
            score = None
            try:
                result = json.loads(output)
                # Accept either 'reasoning' or legacy 'comments'
                reasoning = result.get("reasoning")
                score = result["score"]
            except json.JSONDecodeError:
                print(f"Invalid JSON for proposal {IPTS}: {output}")
                print("===================================")
                print("Could not parse JSON from LLM output")
                print("===================================")

            print(f"Proposal {IPTS} scored {score}")

            # Save review to .md
            reviewer_md_path = f"{folder}/statement_of_resch{IPTS}_{shot}_shot_review.txt"
            with open(reviewer_md_path, 'w') as f:
                f.write(f"Score: {score}\n\n{reasoning}")
                f.write("\n\nFull LLM Output:\n")
                f.write(output)
            # Collect score
            scores.append({"IPTS": IPTS, "score": score})

            print(f"Scored proposal {IPTS}: score {score}, saved review to {reviewer_md_path}")

        except Exception as e:
            print(f"Error scoring proposal {IPTS}: {e}")

    # Save scores to CSV
    csv_path = f"{folder}/proposal_scores_{shot}_shot.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["IPTS", "score"])
        writer.writeheader()
        writer.writerows(scores)
    print(f"Scores saved to {csv_path}")


def score_proposals_parallel(llm_data_folder, llm_model, IPTSs, folder, prompt_file):
    """
    Score proposals in parallel using LLM based on the reviewer prompt.

    Args:
        IPTSs (list): List of proposal numbers
        folder (str): Path to the folder containing the .md files
        prompt_file (str): Path to the prompt file to use for scoring.
    """
    llm_model = llm_model.replace('/', '_')
    print(f"Processing {len(IPTSs)} proposals in parallel")

    # Load the prompt template from the specified file
    with open(prompt_file, 'r') as f:
        content = f.read()

    # Split into system and user prompts
    parts = content.split("# user prompt")
    system_prompt = parts[0].replace("# system prompt", "").strip()
    user_template = parts[1].strip()

    print(f"Loaded prompt template from {prompt_file}")

    # Prepare output directory: llm_data_folder/{llm_model}/review
    review_dir = os.path.join(llm_data_folder, llm_model, "review")
    os.makedirs(review_dir, exist_ok=True)

    def process_proposal(IPTS):
        try:
            md_path = os.path.join(folder, f"statement_of_resch{IPTS}.md")
            # Read the proposal markdown
            with open(md_path, 'r') as f:
                proposal_text = f.read()

            # Fill the user prompt
            user_content = user_template.replace("{proposal_text}", proposal_text)

            # Get LLM response
            output, _ = get_llm_response(user_content, system_prompt)

            # Save the full LLM output only (no parsing)
            review_path = os.path.join(review_dir, f"statement_of_resch{IPTS}_llm_review.txt")
            with open(review_path, 'w') as f:
                f.write(output)

            return f"Saved LLM output for proposal {IPTS} to {review_path}"
        except Exception as e:
            return f"Error scoring proposal {IPTS}: {e}"

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_proposal, IPTS) for IPTS in IPTSs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print(f"All LLM outputs saved to {review_dir}")


def parse_score_output(output):
    """
    Parse the LLM output for scoring results.

    Extracts:
    - Individual dimension scores (R, F, D, T, I, A, DC, CS, P, C)
    - Proposal Type Modifier (PT)
    - Final PPM Score
    - Classification
    - Narrative Summary

    Args:
        output (str): Raw LLM output

    Returns:
        dict: Parsed results containing scores, PPM, classification, summary, and raw output
    """
    raw = output
    output = (output or "").strip()

    result = {
        "scores": {},
        "pt_modifier": None,
        "ppm_score": None,
        "classification": None,
        "summary": None,
        "raw_output": raw,
        "parsed_ok": False
    }

    try:
        # Extract individual dimension scores (R, F, D, T, I, A, DC, CS, P, C)
        dimensions = ['R', 'F', 'D', 'T', 'I', 'A', 'DC', 'CS', 'P', 'C']

        for dim in dimensions:
            # Match patterns like "R: 4", "R = 4", "**R: 4**", "**R = 4**"
            pattern = rf'^[*\s]*{dim}\s*[:=]\s*(\d+(?:\.\d+)?)(?=\D|$)'
            match = re.search(pattern, output, re.MULTILINE)
            if not match:
                # Try to match bolded or markdown style
                pattern_bold = rf'\*\*{dim}\s*[:=]\s*(\d+(?:\.\d+)?)(?=\D|$)'
                match = re.search(pattern_bold, output, re.MULTILINE)
            if match:
                result["scores"][dim] = float(match.group(1))

        # Extract Proposal Type Modifier (PT)
        pt_patterns = [
            r'Proposal Type Modifier\s*\(PT\)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'PT\s*[:=]\s*(\d+(?:\.\d+)?)'
        ]
        for pt_pat in pt_patterns:
            pt_match = re.search(pt_pat, output, re.IGNORECASE)
            if pt_match:
                result["pt_modifier"] = float(pt_match.group(1))
                break

        # Extract Final PPM Score
        # Try multiple patterns and choose the LAST occurrence in the text to avoid
        # picking intermediate calculation lines like "PPM = 1.0 * (...)".
        # Also accept comma decimals (e.g., 4,3) and normalize to dot.
        ppm_patterns = [
            r'Final PPM Score\s*[:=]\s*(\d+(?:[\.,]\d+)?)',
            r'PPM\s*[:=]\s*(\d+(?:[\.,]\d+)?)',
            r'Final.*PPM.*?[:=]\s*(\d+(?:[\.,]\d+)?)',
            r'PPM\s*=\s*(\d+(?:[\.,]\d+)?)',
            r'PPM\s*is\s*(\d+(?:[\.,]\d+)?)',
        ]

        last_ppm_val = None
        last_ppm_pos = -1
        for pattern in ppm_patterns:
            for m in re.finditer(pattern, output, re.IGNORECASE):
                pos = m.start()
                val_str = m.group(1).replace(',', '.')
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if pos >= last_ppm_pos:
                    last_ppm_pos = pos
                    last_ppm_val = val

        if last_ppm_val is not None:
            result["ppm_score"] = last_ppm_val

        # Extract Classification
        class_match = re.search(r'Classification\s*[:=]\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
        if class_match:
            result["classification"] = class_match.group(1).strip()

        # Extract Narrative Summary
        # Look for text after "Narrative Summary" heading
        summary_match = re.search(
            r'Narrative Summary.*?[:=]\s*(.+?)(?:\n\n|\Z)',
            output,
            re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            result["summary"] = summary_match.group(1).strip()

        # Check if we got the essential fields
        if result["ppm_score"] is not None and len(result["scores"]) >= 8:
            result["parsed_ok"] = True
        else:
            print(f"Warning: Incomplete parsing. PPM={result['ppm_score']}, Scores count={len(result['scores'])}")

    except Exception as e:
        print(f"Error parsing score output: {e}")
        result["parsed_ok"] = False

    return result


def process_llm_score_output(llm_data_folder, llm_model):
    """
    Process saved LLM score outputs and generate a CSV file with results.

    Args:
        llm_data_folder (str): Path to the folder containing saved LLM outputs
        llm_model (str): LLM model used for scoring

    Returns:
        str: Path to the generated CSV file
    """
    llm_model_folder = os.path.join(llm_data_folder, llm_model.replace('/', '_'))
    scores_folder = os.path.join(llm_model_folder, 'review')
    print(f"Processing LLM score outputs from {scores_folder}")

    if not os.path.exists(scores_folder):
        print(f"Error: Scores folder not found at {scores_folder}")
        return None

    # Initialize CSV with header
    csv_path = os.path.join(llm_model_folder, "proposal_scores.csv")

    # Define all columns
    fieldnames = [
        "IPTS", "R", "F", "D", "T", "I", "A", "DC", "CS", "P", "C",
        "PT", "PPM_Score", "Classification", "Summary", "llm_model"
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Find all LLM output files in scores_folder
    # Expected format: statement_of_resch{IPTS}_llm_review.txt
    llm_files = [f for f in os.listdir(scores_folder) if f.endswith('_llm_review.txt')]
    print(f"Found {len(llm_files)} LLM score output files")

    failed_count = 0
    success_count = 0

    for llm_file in sorted(llm_files):
        try:
            # Extract IPTS number from filename
            match = re.search(r'statement_of_resch(\d+)_llm_review\.txt$', llm_file)
            if not match:
                print(f"Warning: Could not extract IPTS from filename: {llm_file}")
                continue

            ipts = int(match.group(1))

            # Read LLM output
            file_path = os.path.join(scores_folder, llm_file)
            with open(file_path, 'r') as f:
                output = f.read()

            # Parse output
            parsed = parse_score_output(output)

            if not parsed.get("parsed_ok", False):
                failed_count += 1
                print(f"Failed to parse {llm_file}")

            # Prepare CSV row
            row = {
                "IPTS": ipts,
                "R": parsed["scores"].get("R", ""),
                "F": parsed["scores"].get("F", ""),
                "D": parsed["scores"].get("D", ""),
                "T": parsed["scores"].get("T", ""),
                "I": parsed["scores"].get("I", ""),
                "A": parsed["scores"].get("A", ""),
                "DC": parsed["scores"].get("DC", ""),
                "CS": parsed["scores"].get("CS", ""),
                "P": parsed["scores"].get("P", ""),
                "C": parsed["scores"].get("C", ""),
                "PT": parsed["pt_modifier"] if parsed["pt_modifier"] is not None else "",
                "PPM_Score": parsed["ppm_score"] if parsed["ppm_score"] is not None else "",
                "Classification": parsed["classification"] if parsed["classification"] else "",
                "Summary": parsed["summary"] if parsed["summary"] else "",
                "llm_model": llm_model
            }

            # Append to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            success_count += 1
            print(f"Processed IPTS {ipts}: PPM Score = {parsed['ppm_score']}")

        except Exception as e:
            print(f"Error processing {llm_file}: {e}")
            failed_count += 1

    print("\nProcessing complete:")
    print(f"  Success: {success_count} / {len(llm_files)}")
    print(f"  Failed: {failed_count} / {len(llm_files)}")
    print(f"  Results saved to: {csv_path}")

    return csv_path
