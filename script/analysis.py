import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


# Helper functions
def get_bt_csv_path(llm_model_folder, prompt_type, model):
    """
    Construct path to Bradley-Terry scores CSV file.

    Args:
        folder (str): Path to folder containing LLM output CSVs
        prompt_type (str): Prompt type (e.g., "plain", "rubric", "empty")
        model (str): Model name

    Returns:
        str or None: Path to CSV if it exists, None otherwise
    """
    bt_csv_path = os.path.join(
        llm_model_folder,
        f"{prompt_type}_proposal_comparisons_bt_scores.csv"
    )
    if os.path.exists(bt_csv_path):
        return bt_csv_path
    else:
        print(f"BT score CSV not found: {bt_csv_path}")
        return None


def load_scores(csv_path):
    """
    Load BT scores from CSV file.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        dict: Dictionary mapping IPTS to scores
    """
    scores = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ipts_val = row.get('IPTS', '').strip()
            if not ipts_val:
                continue
            try:
                ipts = int(float(ipts_val))
            except ValueError:
                continue
            score_val = row.get('bt_score', None)
            if score_val is None or score_val == '':
                continue
            try:
                score = float(score_val)
            except ValueError:
                continue
            scores[ipts] = score
    return scores


def load_human_scores(human_csv):
    """
    Load human scores from CSV file.

    Args:
        human_csv (str): Path to human scores CSV

    Returns:
        dict: Dictionary mapping IPTS to human scores
    """
    df = pd.read_csv(human_csv)
    if "IPTS" not in df.columns:
        print("Human CSV must contain IPTS column.")
        return {}

    df["IPTS"] = df["IPTS"].apply(lambda x: int(float(x)))
    human_scores = {}
    for _, row in df.iterrows():
        ipts = row["IPTS"]
        score = row.get("SRCMRating", row.get("score", None))
        if pd.isnull(score):
            continue
        try:
            score_val = float(score)
        except ValueError:
            continue
        human_scores[ipts] = score_val
    return human_scores


def plot_compare_rankings(llm_model_folder, llm_model, prompt_type, human_csv):
    """
    Compare LLM ranking vs human ranking for a specific prompt type.

    Args:
        llm_model_folder (str): Path to folder containing LLM output CSVs
        llm_model (str): Model name
        prompt_type (str): Prompt type to analyze (e.g., "plain", "rubric", "empty")
        human_csv (str): Path to human scores CSV
    """
    # Load LLM scores
    model_csv = get_bt_csv_path(llm_model_folder, prompt_type, llm_model)
    if not model_csv:
        return None

    llm_scores = load_scores(model_csv)

    # Load human scores
    human_scores = load_human_scores(human_csv)

    # Find common IPTS
    common_ipts = sorted(set(llm_scores.keys()) & set(human_scores.keys()))
    if not common_ipts:
        print("No common IPTS between LLM and human scores.")
        return None

    # Assign ranks based on scores (descending)
    llm_sorted = sorted(common_ipts, key=lambda x: llm_scores[x], reverse=True)
    human_sorted = sorted(common_ipts, key=lambda x: human_scores[x], reverse=True)
    llm_rank = {ipts: i + 1 for i, ipts in enumerate(llm_sorted)}
    human_rank = {ipts: i + 1 for i, ipts in enumerate(human_sorted)}

    # Prepare aligned rank lists
    model_ranks = [llm_rank[i] for i in common_ipts]
    human_ranks = [human_rank[i] for i in common_ipts]

    # === Metrics ===
    rho, p_rho = spearmanr(model_ranks, human_ranks)
    tau, p_tau = kendalltau(model_ranks, human_ranks)

    print(f"Comparison on {len(common_ipts)} proposals:")
    print(f"  Spearman ρ  = {rho:.3f} (p = {p_rho:.2e})")
    print(f"  Kendall τ   = {tau:.3f} (p = {p_tau:.2e})")

    # === Plot ===
    model_ranks_array = np.array(model_ranks)
    human_ranks_array = np.array(human_ranks)

    # Calculate absolute rank discrepancy
    rank_discrepancy = np.abs(model_ranks_array - human_ranks_array)

    # Sort indices by discrepancy (descending)
    sorted_indices = np.argsort(rank_discrepancy)[::-1]

    # Calculate correlations for each subset (excluding top k most discrepant)
    max_k = len(common_ipts) // 2
    k_values = list(range(0, max_k + 1))
    rho_values = []
    tau_values = []

    for k in k_values:
        if k == 0:
            # Include all proposals
            subset_model = model_ranks_array
            subset_human = human_ranks_array
        else:
            # Exclude top k most discrepant
            include_indices = sorted_indices[k:]
            subset_model = model_ranks_array[include_indices]
            subset_human = human_ranks_array[include_indices]

        if len(subset_model) > 1:
            rho_k, _ = spearmanr(subset_model, subset_human)
            tau_k, _ = kendalltau(subset_model, subset_human)
            rho_values.append(rho_k)
            tau_values.append(tau_k)
        else:
            rho_values.append(np.nan)
            tau_values.append(np.nan)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*3.3, 3.3*0.9))

    # Set font sizes
    plt.rcParams.update({'font.size': 9})

    # Left subplot: Scatter plot with discrepancy coloring
    scatter = ax1.scatter(human_ranks_array, model_ranks_array,
                          c=rank_discrepancy, cmap='YlOrRd',
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('|LLM Rank - Human Rank|', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax1.plot([1, len(common_ipts)], [1, len(common_ipts)], ls='--', color="gray", linewidth=1)
    ax1.set_xlabel("Human Rank", fontsize=9)
    ax1.set_ylabel("LLM Rank", fontsize=9)
    ax1.set_title(f"LLM ({prompt_type}) vs Human Ranking", fontsize=9)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    # Annotate each point with IPTS
    for i, ipts in enumerate(common_ipts):
        ax1.annotate(str(ipts), (human_ranks_array[i], model_ranks_array[i]),
                     textcoords="offset points", xytext=(-7, 3), ha='left', fontsize=5)

    # Right subplot: Correlation vs k (excluded proposals)
    ax2.plot(k_values, rho_values, marker='o', mfc="None", label='Spearman ρ', linewidth=2, markersize=6)
    ax2.plot(k_values, tau_values, marker='s', mfc="None", label='Kendall τ', linewidth=2, markersize=6)
    ax2.set_xlabel("#Excluded Most Discrepant Proposals (k)", fontsize=9)
    ax2.set_ylabel("Correlation Coefficient", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    # Save plot with prompt type in filename
    plot_path = os.path.join(llm_model_folder, f"{prompt_type}_llm_vs_human_ranking.png")
    plt.savefig(plot_path, dpi=400)
    plt.show()
    plt.close()
    print(f"Ranking comparison plot saved to {plot_path}")

    return {"spearman": rho, "kendall": tau}


def plot_compare_LLM_rankings(llm_model_folder, llm_model, prompt_types=["empty", "plain"]):
    """
    Compare rankings between two LLM outputs (different prompt types) and plot their rank correlation.

    Args:
        llm_model_folder (str): Path to folder containing LLM output CSVs
        llm_model (str): Model name
        prompt_types (list): List of two prompt types to compare (e.g., ["plain", "rubric"])
    """
    if len(prompt_types) != 2:
        print("plot_compare_LLM_rankings requires exactly two prompt types to compare.")
        return None

    # Build CSV paths for each prompt type (using BT scores only)
    csv1 = get_bt_csv_path(llm_model_folder, prompt_types[0], llm_model)
    csv2 = get_bt_csv_path(llm_model_folder, prompt_types[1], llm_model)
    if not csv1 or not csv2:
        print("Missing score CSVs for comparison.")
        return None

    # Load BT scores
    scores1 = load_scores(csv1)
    scores2 = load_scores(csv2)

    # Find common IPTS
    common_ipts = sorted(set(scores1.keys()) & set(scores2.keys()))
    if not common_ipts:
        print("No common IPTS between LLM outputs.")
        return None

    # Assign ranks based on scores (descending)
    sorted1 = sorted(common_ipts, key=lambda x: scores1[x], reverse=True)
    sorted2 = sorted(common_ipts, key=lambda x: scores2[x], reverse=True)
    rank1 = {ipts: i + 1 for i, ipts in enumerate(sorted1)}
    rank2 = {ipts: i + 1 for i, ipts in enumerate(sorted2)}

    ranks1 = [rank1[i] for i in common_ipts]
    ranks2 = [rank2[i] for i in common_ipts]

    # Metrics
    rho, p_rho = spearmanr(ranks1, ranks2)
    tau, p_tau = kendalltau(ranks1, ranks2)

    print(f"LLM ranking comparison on {len(common_ipts)} proposals:")
    print(f"  Spearman ρ  = {rho:.3f} (p = {p_rho:.2e})")
    print(f"  Kendall τ   = {tau:.3f} (p = {p_tau:.2e})")

    # Plot
    ranks1_array = np.array(ranks1)
    ranks2_array = np.array(ranks2)
    rank_discrepancy = np.abs(ranks1_array - ranks2_array)
    sorted_indices = np.argsort(rank_discrepancy)[::-1]
    max_k = len(common_ipts) // 2
    k_values = list(range(0, max_k + 1))
    rho_values = []
    tau_values = []
    for k in k_values:
        if k == 0:
            subset1 = ranks1_array
            subset2 = ranks2_array
        else:
            include_indices = sorted_indices[k:]
            subset1 = ranks1_array[include_indices]
            subset2 = ranks2_array[include_indices]
        if len(subset1) > 1:
            rho_k, _ = spearmanr(subset1, subset2)
            tau_k, _ = kendalltau(subset1, subset2)
            rho_values.append(rho_k)
            tau_values.append(tau_k)
        else:
            rho_values.append(np.nan)
            tau_values.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*3.3, 3.3*0.9))

    # Set font sizes
    plt.rcParams.update({'font.size': 9})

    scatter = ax1.scatter(ranks1_array, ranks2_array,
                          c=rank_discrepancy, cmap='YlOrRd',
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax1, label='|Rank1 - Rank2|')
    cbar.ax.tick_params(labelsize=7)
    ax1.plot([1, len(common_ipts)], [1, len(common_ipts)], ls='--', color="gray", linewidth=1)
    ax1.set_xlabel(f"{prompt_types[0]} LLM Rank", fontsize=9)
    ax1.set_ylabel(f"{prompt_types[1]} LLM Rank", fontsize=9)
    ax1.set_title(f"LLM ({prompt_types[0]}) vs LLM ({prompt_types[1]}) Ranking", fontsize=9)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    for i, ipts in enumerate(common_ipts):
        ax1.annotate(str(ipts), (ranks1_array[i], ranks2_array[i]),
                     textcoords="offset points", xytext=(-7, 3), ha='left', fontsize=7)

    ax2.plot(k_values, rho_values, marker='o', mfc="None", label='Spearman ρ', linewidth=2, markersize=6)
    ax2.plot(k_values, tau_values, marker='s', mfc="None", label='Kendall τ', linewidth=2, markersize=6)
    ax2.set_xlabel("#Excluded Most Discrepant Proposals (k)", fontsize=9)
    ax2.set_ylabel("Correlation Coefficient", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plot_folder = llm_model_folder
    plot_basename = f"llm_{prompt_types[0]}_vs_{prompt_types[1]}_ranking"
    plot_path = os.path.join(plot_folder, f"{plot_basename}.png")
    plt.savefig(plot_path, dpi=400)
    plt.show()
    plt.close()
    print(f"LLM ranking comparison plot saved to {plot_path}")

    return {"spearman": rho, "kendall": tau}


def plot_compare_ranking_variation(llm_model_folder, llm_model, prompt_types, human_csv):
    """
    Plot LLM rankings from multiple prompt types vs human ranking in a single scatter plot.
    Same IPTS across different prompt types are connected, colored by average discrepancy.

    Args:
        llm_model_folder (str): Path to folder containing LLM output CSVs
        llm_model (str): Model name
        prompt_types (list): List of prompt types to compare (e.g., ["empty", "plain", "rubric"])
        human_csv (str): Path to human scores CSV
    """
    # Load human scores
    human_scores = load_human_scores(human_csv)
    if not human_scores:
        return None

    # Load LLM scores for each prompt type
    llm_scores_by_prompt = {}
    for prompt_type in prompt_types:
        csv_path = get_bt_csv_path(llm_model_folder, prompt_type, llm_model)
        if csv_path:
            llm_scores_by_prompt[prompt_type] = load_scores(csv_path)
        else:
            print(f"Skipping prompt type {prompt_type} due to missing CSV.")

    if not llm_scores_by_prompt:
        print("No LLM scores loaded.")
        return None

    # Find common IPTS across all prompt types and human scores
    common_ipts = set(human_scores.keys())
    for prompt_type, scores in llm_scores_by_prompt.items():
        common_ipts &= set(scores.keys())
    common_ipts = sorted(common_ipts)

    if not common_ipts:
        print("No common IPTS across all prompt types and human scores.")
        return None

    print(f"Found {len(common_ipts)} common IPTS across {len(llm_scores_by_prompt)} prompt types and human scores.")

    # Assign human ranks
    human_sorted = sorted(common_ipts, key=lambda x: human_scores[x], reverse=True)
    human_rank = {ipts: i + 1 for i, ipts in enumerate(human_sorted)}

    # Assign LLM ranks for each prompt type
    llm_ranks_by_prompt = {}
    for prompt_type, scores in llm_scores_by_prompt.items():
        sorted_ipts = sorted(common_ipts, key=lambda x: scores[x], reverse=True)
        llm_ranks_by_prompt[prompt_type] = {ipts: i + 1 for i, ipts in enumerate(sorted_ipts)}

    # Calculate average discrepancy per IPTS across all prompt types
    avg_discrepancies = {}
    for ipts in common_ipts:
        discrepancies = []
        for prompt_type in llm_scores_by_prompt.keys():
            llm_rank_val = llm_ranks_by_prompt[prompt_type][ipts]
            human_rank_val = human_rank[ipts]
            discrepancies.append(abs(llm_rank_val - human_rank_val))
        avg_discrepancies[ipts] = np.mean(discrepancies)

    # Calculate metrics for each prompt type
    print("\nCorrelation metrics by prompt type:")
    for prompt_type in llm_scores_by_prompt.keys():
        model_ranks = [llm_ranks_by_prompt[prompt_type][i] for i in common_ipts]
        human_ranks = [human_rank[i] for i in common_ipts]
        rho, p_rho = spearmanr(model_ranks, human_ranks)
        tau, p_tau = kendalltau(model_ranks, human_ranks)
        print(f"  {prompt_type}: Spearman ρ = {rho:.3f} (p = {p_rho:.2e}), Kendall τ = {tau:.3f} (p = {p_tau:.2e})")

    # Create plot
    fig, ax = plt.subplots(figsize=(3.3, 3.3))

    # Set font sizes
    plt.rcParams.update({'font.size': 9})

    # Define colors for each prompt type
    colors = plt.cm.viridis(np.linspace(0, 1, len(prompt_types)))
    prompt_colors = {pt: colors[i] for i, pt in enumerate(prompt_types)}

    # Calculate horizontal offsets for each prompt type to avoid overlap
    n_prompts = len(prompt_types)
    offset_range = 0.3  # Total range of offsets
    offsets = np.linspace(-offset_range/2, offset_range/2, n_prompts)
    prompt_offsets = {pt: offsets[i] for i, pt in enumerate(prompt_types)}

    for ipts in common_ipts:
        human_rank_val = human_rank[ipts]
        llm_rank_vals = [llm_ranks_by_prompt[pt][ipts] for pt in prompt_types]
        human_rank_vals_offset = [human_rank_val + prompt_offsets[pt] for pt in prompt_types]

        # Draw connecting lines
        ax.plot(human_rank_vals_offset, llm_rank_vals,
                "k-", linewidth=1, zorder=1)

    # Plot scatter points for each prompt type
    for prompt_type in prompt_types:
        human_ranks_list = [human_rank[i] + prompt_offsets[prompt_type] for i in common_ipts]
        llm_ranks_list = [llm_ranks_by_prompt[prompt_type][i] for i in common_ipts]

        ax.scatter(human_ranks_list, llm_ranks_list,
                   c=[prompt_colors[prompt_type]],
                   label=prompt_type, alpha=0.7, s=50,
                   edgecolors='black', linewidth=0.5, zorder=2)

    # Add perfect agreement line
    ax.plot([1, len(common_ipts)], [1, len(common_ipts)],
            ls='--', color="gray", linewidth=1, zorder=0)

    ax.set_xlabel("Human Rank", fontsize=9)
    ax.set_ylabel("LLM Rank", fontsize=9)
    ax.set_title(f"LLM Rankings (Multiple Prompts) vs Human Ranking\n{llm_model}", fontsize=9)
    ax.legend(loc='best', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Annotate IPTS (only for points with high average discrepancy to avoid clutter)
    high_discrepancy_threshold = np.percentile(list(avg_discrepancies.values()), 75)
    for ipts in common_ipts:
        if avg_discrepancies[ipts] >= high_discrepancy_threshold:
            # Annotate at the average position of all prompt types
            human_rank_val = human_rank[ipts]
            llm_rank_avg = np.mean([llm_ranks_by_prompt[pt][ipts] for pt in prompt_types])
            ax.annotate(str(ipts), (human_rank_val, llm_rank_avg),
                        textcoords="offset points", xytext=(-7, 3),
                        ha='left', fontsize=7, color='black')

    plt.tight_layout()

    # Save plot
    plot_folder = llm_model_folder
    plot_basename = f"llm_ranking_variation_vs_human_{'_'.join(prompt_types)}"
    plot_path = os.path.join(plot_folder, f"{plot_basename}.png")
    plt.savefig(plot_path, dpi=400)
    plt.show()
    plt.close()
    print(f"\nRanking variation plot saved to {plot_path}")

    return {
        "common_ipts": common_ipts,
        "avg_discrepancies": avg_discrepancies,
        "plot_path": plot_path
    }


def plot_llm_rank_score_publication(llm_model_folder, llm_model, prompt_type, proposal_csv):
    """Plot BT rank vs LLM PPM score, colored by publication counts.

    Creates two side-by-side scatter subplots:
      (1) Color = num_publication
      (2) Color = num_publication_if7

    Each point represents a proposal (IPTS). X-axis is the BT rank (1 = highest
    BT score). Y-axis is the PPM_Score extracted from the LLM scoring pass.

    Args:
        llm_model_folder (str): Folder for this model's outputs.
        llm_model (str): Model name (used for annotation only).
        prompt_type (str): Prompt type whose BT scores to use (e.g. "plain").
        proposal_csv (str): Enriched proposal CSV containing columns:
            IPTS, num_publication, num_publication_if7 (and possibly others).

    Returns:
        dict: { 'plot_path': saved PNG path, 'n_points': number of plotted proposals }

    Example:
        plot_llm_rank_score_publication(
            llm_model_folder,
            llm_model,
            "plain",
            "./data/2021A_enriched.csv"
        )
    """

    # --- Locate BT score CSV robustly ---
    bt_candidates = [
        os.path.join(llm_model_folder, f"{prompt_type}_proposal_comparisons_bt_scores.csv"),
        os.path.join(llm_model_folder, f"{prompt_type}_proposal_comparisons_proposal_bt_scores.csv"),  # current naming pattern
        os.path.join(llm_model_folder, f"{prompt_type}_proposal_comparisons_{llm_model.replace('/', '_')}_bt_scores.csv"),
    ]
    bt_csv = None
    for cand in bt_candidates:
        if os.path.exists(cand):
            bt_csv = cand
            break
    if bt_csv is None:
        print("Error: Could not locate BT score CSV. Tried:")
        for cand in bt_candidates:
            print(f"  {cand}")
        return None

    # --- Locate proposal score CSV produced by process_llm_score_output ---
    score_csv = os.path.join(llm_model_folder, "proposal_scores.csv")
    if not os.path.exists(score_csv):
        print(f"Error: proposal score CSV not found: {score_csv}")
        return None

    # --- Load BT scores ---
    bt_scores = load_scores(bt_csv)  # {IPTS: bt_score}
    if not bt_scores:
        print(f"No BT scores loaded from {bt_csv}")
        # mcolors imported at module level

        return None

    # --- Load LLM PPM scores ---
    df_scores = pd.read_csv(score_csv)
    if "IPTS" not in df_scores.columns or "PPM_Score" not in df_scores.columns:
        print("PPM score CSV must contain IPTS and PPM_Score columns.")
        return None
    df_scores["IPTS"] = df_scores["IPTS"].apply(lambda x: int(float(x)))

    # --- Load enriched proposal CSV for publication counts ---
    if not os.path.exists(proposal_csv):
        print(f"Error: proposal CSV not found: {proposal_csv}")
        return None
    df_prop = pd.read_csv(proposal_csv)
    if "IPTS" not in df_prop.columns:
        print("Enriched proposal CSV must contain IPTS column.")
        return None
    df_prop["IPTS"] = df_prop["IPTS"].apply(lambda x: int(float(x)))

    # Publication columns (handle absent gracefully)
    pub_col = "num_publication"
    pub_if7_col = "num_publication_if7"
    if pub_col not in df_prop.columns:
        print(f"Warning: Column {pub_col} not found; filling zeros.")
        df_prop[pub_col] = 0
    if pub_if7_col not in df_prop.columns:
        print(f"Warning: Column {pub_if7_col} not found; filling zeros.")
        df_prop[pub_if7_col] = 0

    # --- Merge all data ---
    df_bt = pd.DataFrame([
        {"IPTS": ipts, "bt_score": score} for ipts, score in bt_scores.items()
    ])

    df_merge = df_bt.merge(df_scores[["IPTS", "PPM_Score"]], on="IPTS", how="left")
    df_merge = df_merge.merge(df_prop[["IPTS", pub_col, pub_if7_col]], on="IPTS", how="left")

    # Drop rows missing PPM_Score
    before_drop = len(df_merge)
    df_merge = df_merge.dropna(subset=["PPM_Score"])  # ensures numeric
    after_drop = len(df_merge)
    if after_drop == 0:
        print("No proposals with both BT score and PPM_Score available.")
        return None
    if after_drop < before_drop:
        print(f"Dropped {before_drop - after_drop} proposals missing PPM_Score.")

    # Ensure numeric types
    df_merge["PPM_Score"] = pd.to_numeric(df_merge["PPM_Score"], errors="coerce")
    df_merge[pub_col] = pd.to_numeric(df_merge[pub_col], errors="coerce").fillna(0)
    df_merge[pub_if7_col] = pd.to_numeric(df_merge[pub_if7_col], errors="coerce").fillna(0)

    # --- Compute BT rank as unique sequential order (avoid stacking on ties) ---
    # Sort by bt_score desc and IPTS asc as deterministic tie-breaker, then assign unique ranks
    df_merge = df_merge.sort_values(["bt_score", "IPTS"], ascending=[False, True])
    df_merge["bt_rank"] = np.arange(1, len(df_merge) + 1, dtype=float)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3.3, 3.3))
    plt.rcParams.update({'font.size': 9})

    # Common scatter kwargs
    scatter_kwargs = dict(alpha=0.75, s=55, edgecolors='black', linewidth=0.4)

    # First subplot: color by num_publication
    sc1 = axes[0].scatter(
        df_merge["bt_rank"], df_merge["PPM_Score"],
        c=df_merge[pub_col], cmap='viridis', **scatter_kwargs
    )
    axes[0].set_xlabel("BT Rank (1 = highest)", fontsize=9)
    axes[0].set_ylabel("PPM Score", fontsize=9)
    axes[0].set_title(f"PPM vs BT Rank\nColor: {pub_col}", fontsize=9)
    axes[0].grid(alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label(pub_col, fontsize=8)
    cbar1.ax.tick_params(labelsize=7)

    # Second subplot: color by num_publication_if7
    sc2 = axes[1].scatter(
        df_merge["bt_rank"], df_merge["PPM_Score"],
        c=df_merge[pub_if7_col], cmap='plasma', **scatter_kwargs
    )
    axes[1].set_xlabel("BT Rank (1 = highest)", fontsize=9)
    axes[1].set_ylabel("PPM Score", fontsize=9)
    axes[1].set_title("PPM vs BT Rank", fontsize=9)
    axes[1].grid(alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label(pub_if7_col, fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    # Annotate IPTS (avoid clutter: annotate if publication count high or low extremes)
    pub_threshold_high = np.percentile(df_merge[pub_col], 80)
    pub_threshold_low = np.percentile(df_merge[pub_col], 20)
    for _, row in df_merge.iterrows():
        if row[pub_col] >= pub_threshold_high or row[pub_col] <= pub_threshold_low:
            axes[0].annotate(str(int(row['IPTS'])), (row['bt_rank'], row['PPM_Score']),
                             textcoords="offset points", xytext=(-6, 3), fontsize=4)
        # Use IF7 thresholds for second subplot
    if len(df_merge) >= 5:
        pub_if7_high = np.percentile(df_merge[pub_if7_col], 80)
        pub_if7_low = np.percentile(df_merge[pub_if7_col], 20)
    else:
        pub_if7_high = df_merge[pub_if7_col].max()
        pub_if7_low = df_merge[pub_if7_col].min()
    for _, row in df_merge.iterrows():
        if row[pub_if7_col] >= pub_if7_high or row[pub_if7_col] <= pub_if7_low:
            axes[1].annotate(str(int(row['IPTS'])), (row['bt_rank'], row['PPM_Score']),
                             textcoords="offset points", xytext=(-6, 3), fontsize=4)

    # Invert x-axis so that rank 1 (best) appears on the right side
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()

    plt.tight_layout()
    plot_path = os.path.join(llm_model_folder, f"{prompt_type}_bt_rank_vs_ppm_publications.png")
    plt.savefig(plot_path, dpi=400)
    plt.show()
    plt.close()
    print(f"BT Rank vs PPM publication plot saved to {plot_path}")

    return {"plot_path": plot_path, "n_points": len(df_merge)}


def plot_llm_score_distribution_by_publication(llm_model_folder, llm_model , proposal_csv):
    """Plot distribution of PPM scores for published vs unpublished proposals.

    Uses the proposal score CSV produced by the LLM scoring pass (expected at
    "<llm_model_folder>/proposal_scores.csv") and an enriched proposal CSV that
    contains publication columns to split proposals into two groups:
      - Published: num_publication > 0 (falls back to 0 if column missing)
      - Unpublished: num_publication == 0

    The plot shows two overlaid histograms:
      - X-axis: PPM score in [0, 5] with bin size 0.5
      - Y-axis: frequency (number of proposals)

    Args:
        llm_model_folder (str): Folder holding LLM outputs, incl. proposal_scores.csv
        llm_model (str): Model name (used for figure title only)
        proposal_csv (str): Enriched proposal CSV with at least columns:
            - IPTS (identifier)
            - num_publication (optional; defaults to 0 if missing)

    Returns:
        dict | None: {
            'plot_path': str,
            'n_published': int,
            'n_unpublished': int,
            'bins': list[float]
        } or None on error.
    """

    # --- Locate inputs ---
    score_csv = os.path.join(llm_model_folder, "proposal_scores.csv")
    if not os.path.exists(score_csv):
        print(f"Error: proposal score CSV not found: {score_csv}")
        return None
    if not os.path.exists(proposal_csv):
        print(f"Error: proposal CSV not found: {proposal_csv}")
        return None

    # --- Load proposal scores (PPM) ---
    df_scores = pd.read_csv(score_csv)
    required_score_cols = {"IPTS", "PPM_Score"}
    if not required_score_cols.issubset(df_scores.columns):
        print(f"Error: {score_csv} must contain columns: {required_score_cols}")
        return None
    # Normalize IPTS to int, PPM to numeric
    df_scores["IPTS"] = df_scores["IPTS"].apply(lambda x: int(float(x)))
    df_scores["PPM_Score"] = pd.to_numeric(df_scores["PPM_Score"], errors="coerce")

    # --- Load proposal metadata for publication info ---
    df_prop = pd.read_csv(proposal_csv)
    if "IPTS" not in df_prop.columns:
        print("Error: Enriched proposal CSV must contain IPTS column.")
        return None
    df_prop["IPTS"] = df_prop["IPTS"].apply(lambda x: int(float(x)))

    pub_col = "num_publication"
    if pub_col not in df_prop.columns:
        print(f"Warning: Column {pub_col} not found; assuming zeros (unpublished).")
        df_prop[pub_col] = 0
    df_prop[pub_col] = pd.to_numeric(df_prop[pub_col], errors="coerce").fillna(0)

    # --- Merge and filter ---
    df = df_scores.merge(df_prop[["IPTS", pub_col]], on="IPTS", how="left")
    # Missing pub info -> treat as 0
    df[pub_col] = df[pub_col].fillna(0)

    # Keep only rows with a valid PPM score
    before = len(df)
    df = df.dropna(subset=["PPM_Score"]).copy()
    after = len(df)
    if after == 0:
        print("No proposals with valid PPM_Score to plot.")
        return None
    if after < before:
        print(f"Dropped {before - after} proposals with missing PPM_Score.")

    # --- Split groups ---
    df["is_published"] = df[pub_col] > 0
    published_scores = df.loc[df["is_published"], "PPM_Score"].dropna().tolist()
    unpublished_scores = df.loc[~df["is_published"], "PPM_Score"].dropna().tolist()

    n_published = len(published_scores)
    n_unpublished = len(unpublished_scores)
    if n_published == 0 and n_unpublished == 0:
        print("No PPM scores available for either group.")
        return None

    # --- Define bins and plot ---
    bins = np.arange(0.0, 5.0 + 0.5, 0.2)

    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(3.3, 3.3))

    plotted_any = False
    # Plot unpublished first so published overlays
    if n_unpublished > 0:
        ax.hist(
            unpublished_scores,
            bins=bins,
            color="#1f77b4",
            alpha=0.6,
            label=f"Unpublished (n={n_unpublished})",
            edgecolor="black",
            linewidth=0.4
        )
        plotted_any = True
    else:
        print("Note: No unpublished proposals to plot.")

    if n_published > 0:
        ax.hist(
            published_scores,
            bins=bins,
            color="#ff7f0e",
            alpha=0.6,
            label=f"Published (n={n_published})",
            edgecolor="black",
            linewidth=0.4
        )
        plotted_any = True
    else:
        print("Note: No published proposals to plot.")

    if not plotted_any:
        print("Nothing to plot after filtering.")
        return None

    ax.set_xlabel("PPM Score (0–5)", fontsize=9)
    ax.set_ylabel("Frequency (number of proposals)", fontsize=9)
    ax.set_title(f"PPM Score Distribution by Publication Status\n{llm_model}", fontsize=9)
    ax.set_xlim(0, 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(
        llm_model_folder,
        "ppm_score_distribution_published_vs_unpublished.png"
    )
    plt.savefig(plot_path, dpi=400)
    plt.show()
    plt.close()
    print(f"PPM score distribution plot saved to {plot_path}")

    return {
        "plot_path": plot_path,
        "n_published": n_published,
        "n_unpublished": n_unpublished,
        "bins": bins.tolist(),
    }
