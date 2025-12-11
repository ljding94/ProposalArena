#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import time
from llm_rank import *
from analysis import *
from pre_process import *
from llm_score import *
from llm_data_analysis import *
from llm_similarity import *


def main():
    # LLM model to use
    llm_model = "google/gemini-2.5-flash-preview-09-2025"
    # llm_model = "deepseek/deepseek-chat-v3.1"
    # llm_model = "qwen/qwen3-next-80b-a3b-thinking"

    # Prompt type to use for comparisons: one of {"plain","rubric","science"}
    prompt_type = "plain"

    # step 1, load proposal number and score file
    # pps_score_csv = "./data/2025B.csv"
    pps_score_csv = "../data/2021B.csv"
    # 2021A: 26522 not found
    # 2021B: 27011 not found
    folder = "../data/2021"
    llm_data_folder = "../data/LLM_data/data_pool"

    # Load the CSV file with proposal numbers and scores
    df = pd.read_csv(pps_score_csv)

    # Remove .1 suffix from IPTS column if present
    df["IPTS"] = df["IPTS"].apply(lambda x: int(float(x)))

    if 0:
        pps_IPTS_csv = "./data/2021A.csv"
        publication_csv = "./data/Publications.csv"
        publication_if7_csv = "./data/Publications_if7.csv"
        enrich_IPTS_table_with_publication(pps_IPTS_csv, publication_csv, publication_if7_csv)
    if 0:

        process_proposals_docstrange(proposal_numbers=df["IPTS"].tolist()[:], folder="./data/2021")
        # used: https://huggingface.co/nanonets/Nanonets-OCR2-3B

    if 0:
        # score_proposals_parallel(llm_data_folder, llm_model, IPTSs=df["IPTS"].tolist()[:], folder="./data/2021", prompt_file="./prompt/prompt_PPM_score.md")
        # IPTSs = [26780, 26083]
        # score_proposals_parallel(llm_data_folder, llm_model, IPTSs=IPTSs, folder="./data/2021", prompt_file="./prompt/prompt_PPM_score.md")
        process_llm_score_output(llm_data_folder, llm_model)

    if 0:
        # prompt_type = "empty" # "plain"  # "plain"  # "rubric"  # "science"
        # prompt_type = "empty"
        prompt_type = "plain"
        # prompt_type = "rubric"
        # prompt_type = "science"

        # compare_proposals_parallel(IPTSs=df["IPTS"].tolist()[:], folder=folder, llm_data_folder=llm_data_folder, llm_model=llm_model, prompt_type=prompt_type)

        print(f"LLM outputs saved to: {llm_data_folder}")
        # Process outputs as a separate step
        process_llm_comparison_outputs(llm_data_folder, llm_model, prompt_type)
        # print(f"Proposal llm_data_folder: {llm_data_folder}")

        # Update input/output paths to use the llm_model subfolder under llm_data_folder
        llm_model_folder = f"{llm_data_folder}/{llm_model.replace('/', '_')}"
        comparison_csv = f"{llm_model_folder}/{prompt_type}_proposal_comparisons.csv"

        # Compute Bradley-Terry (used by LLM arena) scores and plot vs human ratings
        scores, bt_csv_path = calculate_Bradley_Terry_score(comparison_csv)
        # print(f"Bradley-Terry scores saved next to: {bt_csv_path}")

        bt_csv_path = f"{llm_model_folder}/{prompt_type}_proposal_comparisons_{llm_model.replace('/', '_')}_bt_scores.csv"

    if 0:
        llm_model_folder = f"{llm_data_folder}/{llm_model.replace('/', '_')}"
        # analysis
        # plot_compare_rankings(llm_model_folder, llm_model, "empty", "data/2025B.csv")
        # plot_compare_rankings(llm_model_folder, llm_model, "plain", "data/2025B.csv")
        # plot_compare_rankings(llm_model_folder, llm_model, "rubric", "data/2025B.csv")

        # plot_compare_LLM_rankings(llm_model_folder, llm_model, prompt_types=["empty", "plain"])
        # plot_compare_LLM_rankings(llm_model_folder, llm_model, prompt_types=["empty", "rubric"])
        # plot_compare_LLM_rankings(llm_model_folder, llm_model, prompt_types=["plain", "rubric"])

        # plot_compare_ranking_variation(llm_model_folder, llm_model, prompt_types=["empty","plain", "rubric"], human_csv="data/2025B.csv")
        # plot_llm_rank_score_publication(llm_model_folder, llm_model, "plain", "./data/2021A_enriched.csv")
        plot_llm_score_distribution_by_publication(llm_model_folder, llm_model, "./data/2021_enriched.csv")


#################
# pre processing data
#################


def pre_process_proposal_pdf(instrument):
    (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle) = get_input_args_per_beamline(
        instrument
    )
    ipts_list = find_IPST_from_folder(pdf_folder)
    print(f"Found {len(ipts_list)} IPTS in folder {pdf_folder}")
    n = -1
    print(ipts_list[:n])

    results = batch_OCR_pdf(pdf_folder, ipts_list[:n], md_folder, max_size_mb=10)


def pre_process_proposal_table(instrument):
    (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle) = get_input_args_per_beamline(instrument)
    pre_process_proposal_table_csv = clean_and_sort_proposal_table(proposal_orig_xlsx, pdf_folder)
    enrich_proposal_table_with_publication(pre_process_proposal_table_csv, publication_csv, publication_if7_csv)


#################
# LLM ranking
################
# 1 for each run cycle, do pair-wise comparison for proposals (with pdf && has score)
# 2 calculate BT score

starting_N = 18
num_run = 2


def llm_comparison_ranking(instrument, Ni, Nf):
    (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle) = get_input_args_per_beamline(
        instrument
    )
    # 1. find all run cycles
    df = pd.read_csv(enriched_propoal_data_csv)
    df = df.dropna(subset=["Run Cycle"])
    print(df.head())
    print("Columns:", df.columns.tolist())
    run_cycles = sorted(df["Run Cycle"].unique(), key=lambda cycle: (int(cycle.split()[1].split("-")[0]), cycle.split()[1].split("-")[1]))
    print("Sorted run cycles:", run_cycles)
    idx_start = run_cycles.index(starting_cycle)
    print("Index of", starting_cycle, idx_start)
    print("number of run cycles:", len(run_cycles))

    prompt_file = "../prompt/prompt_compare_plain.md"

    for cycle in run_cycles[idx_start + Ni : idx_start + Nf]:
        print("========================================== ")
        print(f"Processing run cycle: {cycle}, index: {run_cycles.index(cycle)}/{len(run_cycles)}")
        print("========================================== ")
        # Filter IPTS for this cycle where has SRC Rating == 1 and with pdf == 1
        cycle_df = df[(df["Run Cycle"] == cycle) & (df["has SRC Rating"] == 1) & (df["with pdf"] == 1)]
        ipts_list = cycle_df["IPTS"].tolist()
        ipts_list = sorted(ipts_list)[:]
        print(f"Using first 3 proposals for testing: {ipts_list}")
        # Create folder for this run cycle
        cycle_folder = os.path.join(llm_output_data_folder, cycle.replace(" ", "_").replace("-", "_"))
        # Run llm_compare_proposal

        llm_compare_proposal(proposal_folder=md_folder, IPTS_list=ipts_list, output_folder=cycle_folder, prompt_file=prompt_file, llm_model=llm_model)

        process_llm_comparison_output(cycle_folder, llm_model)
        comparison_csv = f"{cycle_folder}/{llm_model.split('/')[-1]}/comparison_statistics.csv"
        scores, bt_csv_path =  calculate_proposal_Bradley_Terry_score(comparison_csv)
        plot_win_lost_heatmap(comparison_csv,bt_csv_path)

        print("========================================== ")
        print(f"Finished Processing run cycle: {cycle}, index: {run_cycles.index(cycle)}/{len(run_cycles)}")
        print("========================================== ")


################
# Data analysis
##############
# 1. aggregate oritinal csv data (human score) and llm TB score data
# 2 compare with human ranking
def llm_human_data_analysis(instrument):
    (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle) = get_input_args_per_beamline(
        instrument
    )
    # 1. find all run cycles
    df = pd.read_csv(enriched_propoal_data_csv)
    df = df.dropna(subset=["Run Cycle"])
    run_cycles = sorted(df["Run Cycle"].unique(), key=lambda cycle: (int(cycle.split()[1].split("-")[0]), cycle.split()[1].split("-")[1]))
    print("Sorted run cycles:", run_cycles)
    idx_start = run_cycles.index(starting_cycle)
    print("Index of", starting_cycle, idx_start)
    print("number of run cycles:", len(run_cycles))

    # for cycle in run_cycles[idx_2014B + starting_N : idx_2014B + starting_N + num_run]:
    for cycle in run_cycles[idx_start:]:
        cycle_folder = os.path.join(llm_output_data_folder, cycle.replace(" ", "_").replace("-", "_"))
        comparison_bt_score_csv = f"{cycle_folder}/{llm_model.split('/')[-1]}/comparison_bt_scores.csv"
        aggregate_llm_human_data(comparison_bt_score_csv, enriched_propoal_data_csv)
        aggregated_score_csv = os.path.join(os.path.dirname(comparison_bt_score_csv), "aggregated_score.csv")
        llm_human_rank_correlation(aggregated_score_csv)
        llm_human_publication_metric(aggregated_score_csv)


###########
# Similarity analysis
##########
# NOTE: qwen/qwen3-embedding-8b from deepinfra has zero data retaining policy https://openrouter.ai/qwen/qwen3-embedding-8b https://openrouter.ai/docs/features/zdr
def llm_similarity_analysis(instrument):
    (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle) = get_input_args_per_beamline(
        instrument
    )
    # 1. find all run cycles
    df = pd.read_csv(enriched_propoal_data_csv)
    df = df.dropna(subset=["Run Cycle"])
    run_cycles = sorted(df["Run Cycle"].unique(), key=lambda cycle: (int(cycle.split()[1].split("-")[0]), cycle.split()[1].split("-")[1]))
    print("Sorted run cycles:", run_cycles)
    idx_start = run_cycles.index(starting_cycle)
    print("Index of", starting_cycle, idx_start)
    print("number of run cycles:", len(run_cycles))
    embedding_model = "qwen/qwen3-embedding-8b"

    for cycle in run_cycles[idx_start:]:
        print("========================================== ")
        print(f"Processing run cycle: {cycle}")
        print("========================================== ")
        cycle_folder = os.path.join(llm_output_data_folder, cycle.replace(" ", "_").replace("-", "_"))
        cycle_df = df[(df["Run Cycle"] == cycle) & (df["has SRC Rating"] == 1) & (df["with pdf"] == 1)]
        ipts_list = cycle_df["IPTS"].tolist()
        ipts_list = sorted(ipts_list)[:]
        calc_proposal_embeddings(proposal_folder=md_folder, IPTS_list=ipts_list, output_folder=cycle_folder, embedding_model=embedding_model)
        analyze_proposal_similarity(os.path.join(cycle_folder, embedding_model.split("/")[-1]))


###############
# get input arguments per beamline
#############


def get_input_args_per_beamline(instrument):
    llm_model = "google/gemini-2.5-flash"
    # "google/gemini-2.5-flash-lite"
    # "google/gemini-2.5-pro"
    if instrument == "EQ-SANS":
        # BL6: EQ-SANS
        proposal_orig_xlsx = "../data/BL-6_EQSANS.xlsx"
        pdf_folder = "../data/proposal_pdf_bl6"
        md_folder = "../data/proposal_md_bl6"
        publication_csv = "../data/pub_bl6.csv"
        publication_if7_csv = "../data/pub_bl6_if7.csv"
        enriched_propoal_data_csv = "../data/enriched_pre_processed_BL-6_EQSANS.csv"
        llm_output_data_folder = "../data/LLM_data/EQSANS"
        starting_cycle = "SNS 2014-B"

    elif instrument == "CNCS":
        # BL5: CNCS
        proposal_orig_xlsx = "../data/BL-5_CNCS.xlsx"
        pdf_folder = "../data/proposal_pdf_bl5"
        md_folder = "../data/proposal_md_bl5"
        publication_csv = "../data/pub_bl5.csv"
        publication_if7_csv = "../data/pub_bl5_if7.csv"
        enriched_propoal_data_csv = "../data/enriched_pre_processed_BL-5_CNCS.csv"
        llm_output_data_folder = "../data/LLM_data/CNCS"
        starting_cycle = "SNS 2016-A"

        # error IPTS pdf: 1615

    elif instrument == "POWGEN":
        # BL 11A: POWGEN
        proposal_orig_xlsx = "../data/BL-11A_POWGEN.xlsx"
        pdf_folder = "../data/proposal_pdf_bl11a"
        md_folder = "../data/proposal_md_bl11a"
        publication_csv = "../data/pub_bl11a.csv"
        publication_if7_csv = "../data/pub_bl11a_if7.csv"
        enriched_propoal_data_csv = "../data/enriched_pre_processed_BL-11A_POWGEN.csv"
        llm_output_data_folder = "../data/LLM_data/POWGEN"
        starting_cycle = "SNS 2015-B"

    return (llm_model, proposal_orig_xlsx, pdf_folder, md_folder, publication_csv, publication_if7_csv, enriched_propoal_data_csv, llm_output_data_folder, starting_cycle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["preprocess_pdf", "preprocess_table", "llm_ranking", "llm_human_analysis", "llm_similarity_analysis"]
    parser.add_argument("--mode", choices=choices, required=True)
    args = parser.parse_args()

    # BL6: EQ-SANS
    # BL5: CNCS
    # BL11A: POWGEN
    instrument = "EQ-SANS"
    #instrument = "CNCS"
    #instrument = "POWGEN"

    start_time = time.time()
    if args.mode == "preprocess_pdf":
        pre_process_proposal_pdf(instrument)
    elif args.mode == "preprocess_table":
        pre_process_proposal_table(instrument)
    elif args.mode == "llm_ranking":
        Ni, Nf = 0, 18
        llm_comparison_ranking(instrument, Ni, Nf)
    elif args.mode == "llm_human_analysis":
        llm_human_data_analysis(instrument)
    elif args.mode == "llm_similarity_analysis":
        llm_similarity_analysis(instrument)
    else:
        print(f"Unknown mode: {args.mode}")

    end_time = time.time()
    print(f"Total run time: {end_time - start_time:.2f} seconds")
