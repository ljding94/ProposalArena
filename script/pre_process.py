import pandas as pd  # type: ignore
import os
import base64
import requests  # type: ignore

###############
# Process proposal PDF into markdown files
##############


def find_IPST_from_folder(folder):
    """Find IPTS identifiers from a folder by scanning for `{IPTS}.pdf` files.

    Args:
        folder (str): Path to the directory containing `{IPTS}.pdf` files.

    Returns:
        list[str]: Sorted, de-duplicated list of IPTS basenames (without extension).

    Notes:
        - Only scans the top-level of the folder (non-recursive).
        - Skips hidden files (starting with '.') and macOS resource forks ('._').
        - Accepts any basename; no strict pattern enforcement to support IPTS like '2021A'.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    ipts_set = set()
    try:
        for name in os.listdir(folder):
            # Skip hidden and macOS resource fork files
            if not name or name.startswith('.') or name.startswith('._'):
                continue
            if name.lower().endswith('.pdf'):
                base, _ = os.path.splitext(name)
                # Basic safety: skip empty basenames
                if base:
                    ipts_set.add(base)
    except OSError as e:
        raise RuntimeError(f"Failed to list folder '{folder}': {e}")

    # Sort numerically when possible; non-numeric names (if any) come after
    def _sort_key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return sorted(ipts_set, key=_sort_key)


def OCR_pdf(pdf_path, output_path, max_size_mb=10):
    """
    Convert a PDF to OCR'd text using Mistral-OCR via OpenRouter.
    Saves the OCR'd text to a new file with '_ocr' suffix in the same folder.

    Args:
        pdf_path (str): Path to the PDF file to OCR
        max_size_mb (int): Maximum file size in MB (default: 50MB).
                          Files larger than this will raise an error.

    Returns:
        str: Path to the saved OCR'd text file
    """
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Check file size
    file_size_bytes = os.path.getsize(pdf_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"PDF size: {file_size_mb:.2f} MB")

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"PDF file is too large ({file_size_mb:.2f} MB). "
            f"Maximum supported size is {max_size_mb} MB. "
            f"Consider using a different OCR method for large files (e.g., pymupdf4llm, docstrange, or cloud OCR services)."
        )

    # Encode PDF to base64
    def encode_pdf_to_base64(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")

    print(f"Encoding PDF: {pdf_path}")
    base64_pdf = encode_pdf_to_base64(pdf_path)
    data_url = f"data:application/pdf;base64,{base64_pdf}"

    # Prepare the request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please extract all the text from this document. Provide the complete text content in a well-formatted manner, preserving the document structure as much as possible.",
                },
                {"type": "file", "file": {"filename": os.path.basename(pdf_path), "file_data": data_url}},
            ],
        }
    ]

    # Configure to use mistral-ocr engine
    plugins = [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]  # Use Mistral-OCR for better accuracy
    # Note: We use mistral-ocr plugin — the base model is just a proxy.
    # The actual OCR is done by Mistral-OCR, not Gemini.
    payload = {"model": "google/gemini-2.5-flash-preview-09-2025", "messages": messages, "plugins": plugins}  # Using a capable model for OCR

    print(f"Sending request to OpenRouter API for OCR {pdf_path}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)

        if response.status_code == 200:
            response_data = response.json()
            ocr_text = response_data["choices"][0]["message"]["content"]
        elif response.status_code == 413:
            raise ValueError(
                "PDF file is too large for the API to process. Try reducing max_size_mb parameter or use an alternative OCR method."
            )
        elif response.status_code == 503:
            raise Exception(
                "OpenRouter API is temporarily unavailable (503). Please try again in a few minutes."
            )
        else:
            raise Exception(f"API error: {response.status_code} {response.text}")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The PDF may be too large or complex to process.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")

    # Determine output path
    base, ext = os.path.splitext(pdf_path)

    # Save OCR'd text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    print(f"OCR completed. Saved to: {output_path}")
    return output_path


def batch_OCR_pdf(pdf_folder, IPTS_list, output_folder, max_size_mb=10):
    """Batch OCR multiple PDF files concurrently using the existing OCR_pdf function.

    Each input PDF is expected at ``{pdf_folder}/{IPTS}.pdf`` and the OCR'd
    markdown will be written to ``{output_folder}/{IPTS}.md``.

    Args:
        pdf_folder (str): Folder containing source PDF files.
        IPTS_list (Iterable[str|int]): Collection of IPTS identifiers.
        output_folder (str): Destination folder for markdown outputs.
        max_size_mb (int): Maximum allowed PDF size passed through to OCR_pdf.

    Returns:
        list[dict]: A list of result records with keys:
            - ipts: the IPTS identifier
            - pdf_path: path to source PDF
            - output_path: path to markdown (if success)
            - status: 'ok' or 'error' or 'skipped'
            - error: error message when status == 'error'

    Notes:
        - Creates output folder if it does not exist.
        - Skips processing if output markdown already exists.
        - Processes in a thread pool (I/O bound network calls) for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Normalize IPTS identifiers to strings for path construction
    ipts_items = [str(x) for x in IPTS_list]

    results = []

    def _process_one(ipts: str):
        pdf_path = os.path.join(pdf_folder, f"{ipts}.pdf")
        output_path = os.path.join(output_folder, f"{ipts}.md")

        # Skip if already processed
        if os.path.exists(output_path):
            return {
                "ipts": ipts,
                "pdf_path": pdf_path,
                "output_path": output_path,
                "status": "skipped",
                "error": None,
            }

        # Check existence
        if not os.path.isfile(pdf_path):
            return {
                "ipts": ipts,
                "pdf_path": pdf_path,
                "output_path": None,
                "status": "error",
                "error": "PDF not found",
            }

        try:
            # Delegate to OCR_pdf
            OCR_pdf(pdf_path, output_path, max_size_mb=max_size_mb)
            return {
                "ipts": ipts,
                "pdf_path": pdf_path,
                "output_path": output_path,
                "status": "ok",
                "error": None,
            }
        except Exception as e:
            return {
                "ipts": ipts,
                "pdf_path": pdf_path,
                "output_path": None,
                "status": "error",
                "error": str(e),
            }

    # Use a modest pool size to avoid rate limiting / resource exhaustion
    max_workers = min(8, max(1, len(ipts_items)))
    print(f"Starting batch OCR for {len(ipts_items)} PDFs with up to {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_one, ipts): ipts for ipts in ipts_items}
        for fut in as_completed(future_map):
            res = fut.result()
            results.append(res)
            status = res["status"]
            ipts = res["ipts"]
            if status == "ok":
                print(f"[OK] IPTS {ipts} -> {res['output_path']}")
            elif status == "skipped":
                print(f"[SKIP] IPTS {ipts} already processed -> {res['output_path']}")
            else:
                print(f"[ERROR] IPTS {ipts}: {res['error']}")

    # Sort results by original order for deterministic return
    order_index = {ipts: i for i, ipts in enumerate(ipts_items)}
    results.sort(key=lambda r: order_index.get(r["ipts"], 0))

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"Batch OCR complete: {ok_count} ok, {skip_count} skipped, {err_count} errors.")
    return results


###################
# pre process proposal table data
##################

# 1. clean up original proposal xlsx, in to useful csv file
# should contains,  IPTS (int), SRC Ranking(int, need filtering), SRC Rating(float), Run Cycle (str), Submission Status(str), Instrument(str), Title(str), Abstract(str), Act Start Dte(str), Act End Dte(str), Comment to PI(str), Title(str), Proposal Type.

# 2. enrich proposal table with publication data, based on publication.csv and publication_if7.csv: add columns num_publication, num_publication_if7, publication_year_list( list of publication years)

def clean_and_sort_proposal_table(proposal_orig_xlsx, pdf_folder):
    df = pd.read_excel(proposal_orig_xlsx)

    # Required columns
    required_columns = [
        'IPTS',
        'SRC Rating',
        'Run Cycle',
        'Instrument',
        'Instrument Status',
        'Title',
        'Act Start Dte',
        'Act End Dte',
        'Comment to PI',
        'Proposal Type'
    ]

    # Check if all columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_columns].copy()

    # Convert types
    df['IPTS'] = df['IPTS'].astype(str).str.split('.').str[0]
    df['SRC Rating'] = pd.to_numeric(df['SRC Rating'], errors='coerce')
    # Reformat SRC Rating: divide by 100 if > 100
    df['SRC Rating'] = df['SRC Rating'].where(df['SRC Rating'] <= 100, df['SRC Rating'] / 100)

    # Add has SRC Rating column
    df['has SRC Rating'] = (df['SRC Rating'].notna() & (df['SRC Rating'] > 0)).astype(int)

    # Add with_pdf column
    ipts_with_pdf = set(find_IPST_from_folder(pdf_folder))
    df['with pdf'] = df['IPTS'].isin(ipts_with_pdf).astype(int)

    # Strip string columns
    str_columns = ['Run Cycle', 'Instrument', 'Instrument Status', 'Title', 'Act Start Dte', 'Act End Dte', 'Comment to PI', 'Proposal Type']
    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Clean up 'Comment to PI' by removing '_x000E_' artifacts
    df['Comment to PI'] = df['Comment to PI'].str.replace('_x000E_', '')

    # Add experiment finished column after stripping
    df['experiment finished'] = ((df['Act End Dte'] != 'nan') & (df['Act End Dte'] != '') & (df['Act End Dte'] != 'NaT')).astype(int)

    # Reorder columns to place 'has SRC Rating' after 'IPTS', 'experiment finished' after 'has SRC Rating', and 'with pdf' after 'experiment finished'
    cols = df.columns.tolist()
    ipts_idx = cols.index('IPTS')
    has_src_idx = cols.index('has SRC Rating')
    exp_fin_idx = cols.index('experiment finished')
    with_pdf_idx = cols.index('with pdf')
    # Move has SRC Rating to after IPTS
    cols.insert(ipts_idx + 1, cols.pop(has_src_idx))
    # Move experiment finished to after has SRC Rating
    cols.insert(ipts_idx + 2, cols.pop(exp_fin_idx))
    # Move with pdf to after experiment finished
    cols.insert(ipts_idx + 3, cols.pop(with_pdf_idx))
    df = df[cols]

    # Save to csv
    base_name = os.path.splitext(os.path.basename(proposal_orig_xlsx))[0]
    output_path = os.path.join(os.path.dirname(proposal_orig_xlsx), f"pre_processed_{base_name}.csv")
    df.to_csv(output_path, index=False)

    print(f"Cleaned and sorted proposal table saved to {output_path}")

    return output_path


def enrich_proposal_table_with_publication(pre_process_proposal_table_csv, publication_csv, publication_if7_csv):
    # Load tables if they are file paths
    input_is_path = isinstance(pre_process_proposal_table_csv, str)

    '''
    prompt:
    we should improve the enrich_proposal_table_with_publication function, in addition to the num_publication and num_publication_if7, which only account for how many publications associated with one proposal, we should also calculated discounted_num_publication and discounted_num_publicationif7, such that for publication with more than one associated proposals, we count 1/n for each proposal, where n is the number of associated proposals for that publication.
    '''

    def safe_read_csv(path):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")

    if input_is_path:
        pps_df = safe_read_csv(pre_process_proposal_table_csv)
    else:
        pps_df = pre_process_proposal_table_csv.copy()
    if isinstance(publication_csv, str):
        pub_df = safe_read_csv(publication_csv)
    else:
        pub_df = publication_csv.copy()
    if isinstance(publication_if7_csv, str):
        pub_if7_df = safe_read_csv(publication_if7_csv)
    else:
        pub_if7_df = publication_if7_csv.copy()
    from collections import Counter, defaultdict
    # Helper to count IPTS occurrences

    def count_ipts(df):
        ipts_counter = Counter()
        for val in df["IPTS"].dropna():
            if not isinstance(val, str):
                continue
            for ipts in [x.strip() for x in val.split(",") if x.strip()]:
                ipts_counter[ipts] += 1
        return dict(ipts_counter)

    pub_counts = count_ipts(pub_df)
    pub_if7_counts = count_ipts(pub_if7_df)
    # Calculate discounted publications
    discounted_pub = defaultdict(float)
    for _, row in pub_df.iterrows():
        ipts_str = str(row["IPTS"])
        if ipts_str == 'nan' or not ipts_str.strip():
            continue
        ipts_list = list(set([x.strip() for x in ipts_str.split(",") if x.strip()]))
        n = len(ipts_list)
        if n > 0:
            for ipts in ipts_list:
                discounted_pub[ipts] += 1.0 / n

    discounted_pub_if7 = defaultdict(float)
    for _, row in pub_if7_df.iterrows():
        ipts_str = str(row["IPTS"])
        if ipts_str == 'nan' or not ipts_str.strip():
            continue
        ipts_list = list(set([x.strip() for x in ipts_str.split(",") if x.strip()]))
        n = len(ipts_list)
        if n > 0:
            for ipts in ipts_list:
                discounted_pub_if7[ipts] += 1.0 / n

    # Find IPTS in discounted_pub but not in pps_df
    pps_ipts_set = set(pps_df['IPTS'].astype(str))
    extra_pub_ipts = set(discounted_pub.keys()) - pps_ipts_set
    if extra_pub_ipts:
        print(f"Total IPTS in discounted_pub but not in pps_df: {len(extra_pub_ipts)} - {sorted(extra_pub_ipts)}")
    extra_pub_if7_ipts = set(discounted_pub_if7.keys()) - pps_ipts_set
    if extra_pub_if7_ipts:
        print(f"Total IPTS in discounted_pub_if7 but not in pps_df: {len(extra_pub_if7_ipts)} - {sorted(extra_pub_if7_ipts)}")
    # Find rows in pub_df where none of the IPTS are in pps_df
    missing_pub_rows = []
    for idx, row in pub_df.iterrows():
        ipts_str = str(row["IPTS"])
        if ipts_str == 'nan' or not ipts_str.strip():
            continue
        ipts_list = list(set([x.strip() for x in ipts_str.split(",") if x.strip()]))
        if not any(ipts in pps_ipts_set for ipts in ipts_list):
            missing_pub_rows.append(row)
    if missing_pub_rows:
        print(f"Rows in pub_df with no IPTS in pps_df: {len(missing_pub_rows)}")
        for row in missing_pub_rows:
            print(f"  IPTS: {row['IPTS']}")
    # Similarly for pub_if7_df
    missing_pub_if7_rows = []
    for idx, row in pub_if7_df.iterrows():
        ipts_str = str(row["IPTS"])
        if ipts_str == 'nan' or not ipts_str.strip():
            continue
        ipts_list = list(set([x.strip() for x in ipts_str.split(",") if x.strip()]))
        if not any(ipts in pps_ipts_set for ipts in ipts_list):
            missing_pub_if7_rows.append(row)
    if missing_pub_if7_rows:
        print(f"Rows in pub_if7_df with no IPTS in pps_df: {len(missing_pub_if7_rows)}")
        for row in missing_pub_if7_rows:
            print(f"  IPTS: {row['IPTS']}")
    # Map counts to pps_df
    pps_df["num_publication"] = pps_df["IPTS"].astype(str).map(pub_counts).fillna(0).astype(int)
    pps_df["num_publication_if7"] = pps_df["IPTS"].astype(str).map(pub_if7_counts).fillna(0).astype(int)
    pps_df["discounted_num_publication"] = pps_df["IPTS"].astype(str).map(discounted_pub).fillna(0).round(2)
    pps_df["discounted_num_publication_if7"] = pps_df["IPTS"].astype(str).map(discounted_pub_if7).fillna(0).round(2)
    # Save to file if input was a file path
    if input_is_path:
        dir_path = os.path.dirname(pre_process_proposal_table_csv)
        filename = os.path.basename(pre_process_proposal_table_csv)
        base, ext = os.path.splitext(filename)
        enriched_filename = f"enriched_{base}{ext}"
        enriched_path = os.path.join(dir_path, enriched_filename)
        pps_df.to_csv(enriched_path, index=False)
        print(f"Enriched table saved to {enriched_path}")
    print(pps_df)
    return pps_df
