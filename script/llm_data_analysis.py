import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import spearmanr


def aggregate_llm_human_data(comparison_bt_score_csv, enriched_propoal_data_csv):
    # Read the Bradley-Terry scores CSV
    bt_df = pd.read_csv(comparison_bt_score_csv)

    # Read the enriched proposal data CSV
    enriched_df = pd.read_csv(enriched_propoal_data_csv)

    # Get unique run cycles from bt_df
    run_cycles = bt_df['run_cycle'].unique()

    # Filter enriched_df to only include rows with matching run cycles
    enriched_df = enriched_df[enriched_df['Run Cycle'].isin(run_cycles)]

    # Select the columns to merge from enriched data
    columns_to_merge = [
        'SRC Rating',
        'experiment finished',
        'num_publication',
        'num_publication_if7',
        'discounted_num_publication',
        'discounted_num_publication_if7'
    ]

    # Merge the dataframes on 'IPTS' and 'run_cycle' (bt_df) with 'Run Cycle' (enriched_df)
    merged_df = bt_df.merge(enriched_df[['IPTS', 'Run Cycle'] + columns_to_merge],
                            left_on=['IPTS', 'run_cycle'],
                            right_on=['IPTS', 'Run Cycle'],
                            how='left')

    # Drop the redundant 'Run Cycle' column since it's the same as 'run_cycle'
    merged_df = merged_df.drop(columns=['Run Cycle'])

    # Check for repeated IPTS numbers
    duplicates = merged_df[merged_df.duplicated('IPTS', keep=False)]
    if not duplicates.empty:
        print("Warning: Repeated IPTS numbers found after merge:")
        print(duplicates[['IPTS', 'run_cycle']].drop_duplicates())

    # Remove duplicate IPTS rows, keeping the first occurrence
    len_before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset='IPTS', keep='first')
    len_after = len(merged_df)
    if len_before > len_after:
        print(f"Removed {len_before - len_after} duplicate IPTS rows.")

    # Create the output path for the new aggregated CSV
    output_dir = os.path.dirname(comparison_bt_score_csv)
    aggregated_csv_path = os.path.join(output_dir, 'aggregated_score.csv')

    # Write the merged dataframe to the new CSV
    merged_df.to_csv(aggregated_csv_path, index=False)

    print(f"Aggregated data saved to {aggregated_csv_path}")


# TODO: 1. normalize the rank by dividing total number of proposal
# TODO: 2, normalize the color bar accordingly
# TODO:3 k become percentage of excluded poposals


def llm_human_rank_correlation(aggregated_score_csv):
    df = pd.read_csv(aggregated_score_csv)

    # Sort for LLM_rank: bt_score desc, SRC Rating desc, IPTS asc
    df_llm = df.sort_values(by=['bt_score', 'SRC Rating', 'IPTS'], ascending=[False, False, True])
    df_llm['LLM_rank'] = range(1, len(df_llm) + 1)

    # Sort for Human_Rank: SRC Rating desc, bt_score desc, IPTS asc
    df_human = df.sort_values(by=['SRC Rating', 'bt_score', 'IPTS'], ascending=[False, False, True])
    df_human['Human_Rank'] = range(1, len(df_human) + 1)

    # Identify ties
    ties_llm = []
    for _, group in df_llm.groupby('bt_score'):
        if len(group) > 1:
            ties_llm.extend(group['IPTS'].tolist())
    ties_llm_set = set(ties_llm)

    ties_human = []
    for _, group in df_human.groupby('SRC Rating'):
        if len(group) > 1:
            ties_human.extend(group['IPTS'].tolist())
    ties_human_set = set(ties_human)

    # Merge the ranks back to the original df
    df = df.merge(df_llm[['IPTS', 'LLM_rank']], on='IPTS')
    df = df.merge(df_human[['IPTS', 'Human_Rank']], on='IPTS')

    # Normalize ranks by dividing by total number of proposals
    N = len(df)
    df['LLM_rank_normalized'] = df['LLM_rank'] / N
    df['Human_rank_normalized'] = df['Human_Rank'] / N

    # Normalize scores for coloring markers
    df['SRC_normalized'] = (df['SRC Rating'] - df['SRC Rating'].min()) / (df['SRC Rating'].max() - df['SRC Rating'].min())
    df['bt_normalized'] = (df['bt_score'] - df['bt_score'].min()) / (df['bt_score'].max() - df['bt_score'].min())

    # Define tiers for SRC Rating
    q1, q2 = df['SRC Rating'].quantile([1/3, 2/3])

    def get_color(rating):
        if rating < q1:
            return 'red'
        elif rating < q2:
            return 'orange'
        else:
            return 'green'

    # Compute Spearman's rank correlation
    correlation = df[['LLM_rank', 'Human_Rank']].corr(method='spearman').iloc[0, 1]
    print(f"Spearman's rank correlation between LLM_rank and Human_Rank: {correlation}")

    # Save the df with ranks
    output_dir = os.path.dirname(aggregated_score_csv)
    ranked_csv_path = os.path.join(output_dir, 'ranked_scores.csv')
    df.to_csv(ranked_csv_path, index=False)
    print(f"Ranked data saved to {ranked_csv_path}")

    # Plotting
    llm_ranks = df['LLM_rank'].tolist()
    human_ranks = df['Human_Rank'].tolist()
    llm_ranks_normalized = df['LLM_rank_normalized'].tolist()
    human_ranks_normalized = df['Human_rank_normalized'].tolist()
    ipts_list = df['IPTS'].tolist()

    # Metrics
    rho, p_rho = spearmanr(llm_ranks, human_ranks)

    print(f"  Spearman ρ  = {rho:.3f} (p = {p_rho:.2e})")

    # Plot
    llm_ranks_array = np.array(llm_ranks)
    human_ranks_array = np.array(human_ranks)
    rank_discrepancy = np.abs(df['LLM_rank_normalized'] - df['Human_rank_normalized'])
    sorted_indices = np.argsort(rank_discrepancy)[::-1]
    max_k = len(ipts_list) // 2
    k_values = list(range(0, max_k + 1))
    rho_values = []
    for k in k_values:
        if k == 0:
            subset_llm = llm_ranks_array
            subset_human = human_ranks_array
        else:
            include_indices = sorted_indices[k:]
            subset_llm = llm_ranks_array[include_indices]
            subset_human = human_ranks_array[include_indices]
        if len(subset_llm) > 1:
            rho_k, _ = spearmanr(subset_llm, subset_human)
            rho_values.append(rho_k)
        else:
            rho_values.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*3.3, 3.3*0.9))

    # Set font sizes
    plt.rcParams.update({'font.size': 9})

    scatter = ax1.scatter(human_ranks_normalized, llm_ranks_normalized,
                          c=rank_discrepancy, cmap='YlOrRd',
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5, vmin=0, vmax=1)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Normalized Rank Discrepancy', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    ax1.plot([0, 1], [0, 1], ls='--', color="gray", linewidth=1)
    ax1.set_xlabel("Normalized Human Rank", fontsize=9)
    ax1.set_ylabel("Normalized LLM Rank", fontsize=9)
    ax1.set_title("Normalized LLM vs Human Ranking", fontsize=9)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    for i, ipts in enumerate(ipts_list):
        ax1.annotate(str(ipts), (human_ranks_normalized[i], llm_ranks_normalized[i]),
                     textcoords="offset points", xytext=(-7, 3), ha='left', fontsize=5)

    # Plot ties markers
    ties_llm_x = [human_ranks_normalized[i] for i, ipts in enumerate(ipts_list) if ipts in ties_llm_set]
    ties_llm_y = [llm_ranks_normalized[i] for i, ipts in enumerate(ipts_list) if ipts in ties_llm_set]
    ax1.scatter(ties_llm_x, ties_llm_y, marker='+', color='black', s=50, edgecolors='black', linewidth=0.5)

    ties_human_x = [human_ranks_normalized[i] for i, ipts in enumerate(ipts_list) if ipts in ties_human_set]
    ties_human_y = [llm_ranks_normalized[i] for i, ipts in enumerate(ipts_list) if ipts in ties_human_set]
    ax1.scatter(ties_human_x, ties_human_y, marker='x', color='black', s=50, edgecolors='black', linewidth=0.5)

    k_percentages = [k / N * 100 for k in k_values]
    ax2.plot(k_percentages, rho_values, marker='o', mfc="None", label='Spearman ρ', linewidth=2, markersize=6)
    ax2.set_xlabel("Percentage of Excluded Most Discrepant Proposals (%)", fontsize=9)
    ax2.set_ylabel("Correlation Coefficient", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rank_correlation_plot.png')
    plt.savefig(plot_path, dpi=500)
    #plt.show()
    plt.close()
    print(f"Ranking correlation plot saved to {plot_path}")

#  calculated publication matric
def llm_human_publication_metric(aggregated_score_csv):
    """
    let's implement this llm_human_publication_metric function, where we will firstly find the llm_rank and human_rank using the same method in the llm_human_rank funciton,, then we will calculate the publication metric for each ranking, defined by sum((1-rank/total_number)*num_publication)/sum(num_publication), and we should do the same for num_publication_if7, discounted_num_publication, discounted_num_publication_if7
    finally, we can calculate these 2 x 4 number for each sub rank (with increasing number of excluded most discrepant proposals), and plot them accordingly.
    """
    df = pd.read_csv(aggregated_score_csv)

    # Filter by experiment_finished
    df = df[df['experiment finished'] == 1]

    # Sort for LLM_rank: bt_score desc, SRC Rating desc, IPTS asc
    df_llm = df.sort_values(by=['bt_score', 'SRC Rating', 'IPTS'], ascending=[False, False, True])
    df_llm['LLM_rank'] = range(1, len(df_llm) + 1)

    # Sort for Human_Rank: SRC Rating desc, bt_score desc, IPTS asc
    df_human = df.sort_values(by=['SRC Rating', 'bt_score', 'IPTS'], ascending=[False, False, True])
    df_human['Human_Rank'] = range(1, len(df_human) + 1)

    # Merge the ranks back to the original df
    df = df.merge(df_llm[['IPTS', 'LLM_rank']], on='IPTS')
    df = df.merge(df_human[['IPTS', 'Human_Rank']], on='IPTS')

    # Normalize ranks
    N = len(df)
    df['LLM_rank_normalized'] = df['LLM_rank'] / N
    df['Human_rank_normalized'] = df['Human_Rank'] / N

    # Compute rank discrepancy
    df['rank_discrepancy'] = np.abs(df['LLM_rank_normalized'] - df['Human_rank_normalized'])
    sorted_indices = np.argsort(df['rank_discrepancy'])[::-1]  # most discrepant first

    # Publication columns
    pub_cols = ['discounted_num_publication', 'discounted_num_publication_if7']

    # Rankings
    rankings = ['LLM', 'Human']
    rank_cols = {'LLM': 'LLM_rank', 'Human': 'Human_Rank'}

    # Max k
    max_k = len(df) // 2
    k_values = list(range(0, max_k + 1))

    # Store metrics
    metrics_data = []

    for k in k_values:
        subset_indices = sorted_indices[k:]
        subset_df = df.iloc[subset_indices].copy()

        for ranking in rankings:
            rank_col = rank_cols[ranking]
            for pub_col in pub_cols:
                total_pub = subset_df[pub_col].sum()
                if total_pub > 0:
                    metric = ((1 - subset_df[rank_col] / N) * subset_df[pub_col]).sum() / total_pub
                else:
                    metric = 0
                metrics_data.append({
                    'k': k,
                    'ranking': ranking,
                    'pub_col': pub_col,
                    'metric': metric
                })

    # Convert to DataFrame for plotting
    metrics_df = pd.DataFrame(metrics_data)

    # Plot
    fig, ax = plt.subplots(figsize=(3.3, 3.3*0.8))
    plt.rcParams.update({'font.size': 9})

    for ranking in rankings:
        linestyle = '-' if ranking == 'Human' else '--'
        color = 'red' if ranking == 'Human' else 'blue'
        for pub_col in pub_cols:
            subset = metrics_df[(metrics_df['ranking'] == ranking) & (metrics_df['pub_col'] == pub_col)]
            marker = 's' if 'num_publication' in pub_col and 'if7' not in pub_col else 'o'
            markerfacecolor = 'none'
            markeredgecolor = color
            ax.plot(subset['k'] / N * 100, subset['metric'], marker=marker, label=f'{ranking} {pub_col}', color=color, linestyle=linestyle, linewidth=2, markersize=6, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)

    ax.set_xlabel("Percentage of Excluded Most Discrepant Proposals (%)", fontsize=9)
    ax.set_ylabel("Publication Metric", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='both', labelsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.dirname(aggregated_score_csv)
    plot_path = os.path.join(output_dir, 'publication_metric_plot.png')
    plt.savefig(plot_path, dpi=500)
    #plt.show()
    plt.close()
    print(f"Publication metric plot saved to {plot_path}")

    # Optionally save metrics
    metrics_csv_path = os.path.join(output_dir, 'publication_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Publication metrics saved to {metrics_csv_path}")
