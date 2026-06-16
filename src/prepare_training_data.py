import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


# ==========================================
# 1. SET YOUR FOLDERS
# ==========================================
# Replace these placeholder paths with your actual local paths before running.
source_folder = r"path/to/source/dicom_folder"
output_folder = r"path/to/output/png_folder"
output_csv = r"path/to/output/metadata_database.csv"

IMAGE_SIZE = 1024
NUMBER_TO_SAMPLE = 5000


# ==========================================
# 2. PRINT CSV HEADERS
# ==========================================
def print_csv_headers():
    headers = [
        "filename",
        "relative_path",
        "patient_id",
        "study_uid",
        "laterality",
        "view_position",
        "patient_age",
        "manufacturer",
        "monochrome_type",
        "rows_original",
        "cols_original"
    ]
    print("\nCSV headers:")
    print(",".join(headers))
    print()


# ==========================================
# 3. THE MULTIPROCESSING WORKER (STRICT GATE + HIGH CONTRAST)
# ==========================================
def process_single_file(args):
    dicom_path, save_path, rel_path, filename = args
    try:
        dcm = pydicom.dcmread(dicom_path)

        # --- PART A: Strict Quality Gate (No Rescues) ---
        try:
            img = apply_voi_lut(dcm.pixel_array, dcm)
        except Exception:
            return False, None, f"Corrupted VOI LUT Tags: {dicom_path}"

        # If VOI LUT produced a perfectly flat gray/black square, reject and log.
        if np.max(img) == np.min(img):
            return False, None, f"Flat Image Output (Garbage Tags): {dicom_path}"

        # --- PART B: Image Processing ---
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            img = np.amax(img) - img

        # The 1-99% Percentile Clip (Chops off metal markers and air)
        p_min = np.percentile(img, 1)
        p_max = np.percentile(img, 99)
        img = np.clip(img, p_min, p_max)

        # Safe 0-255 Normalization
        img = img - np.min(img)
        if np.max(img) != 0:
            img = (img / np.max(img)) * 255.0

        img = np.uint8(img)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # --- PART C: Extract Metadata ---
        data = {
            "filename": filename,
            "relative_path": rel_path,
            "patient_id": getattr(dcm, 'PatientID', 'Unknown'),
            "study_uid": getattr(dcm, 'StudyInstanceUID', 'Unknown'),
            "laterality": getattr(dcm, 'ImageLaterality', 'Unknown'),
            "view_position": getattr(dcm, 'ViewPosition', 'Unknown'),
            "patient_age": getattr(dcm, 'PatientAge', 'Unknown'),
            "manufacturer": getattr(dcm, 'Manufacturer', 'Unknown'),
            "monochrome_type": getattr(dcm, 'PhotometricInterpretation', 'Unknown'),
            "rows_original": getattr(dcm, 'Rows', 'Unknown'),
            "cols_original": getattr(dcm, 'Columns', 'Unknown')
        }

        # --- PART D: Save File ---
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        write_ok = cv2.imwrite(save_path, img_resized)
        if not write_ok:
            return False, None, f"Failed to write PNG: {save_path}"

        return True, data, None

    except Exception as e:
        return False, None, f"Path: {dicom_path} | Error: {e}"


# ==========================================
# 4. RUN THE FAST PIPELINE
# ==========================================
def run_pipeline():
    os.makedirs(output_folder, exist_ok=True)

    # --- Deduplication Logic ---
    existing_records = set()
    if os.path.exists(output_csv):
        print(f"Checking existing database at {output_csv} for deduplication...")
        try:
            try:
                existing_df = pd.read_csv(output_csv, usecols=['relative_path'])
            except ValueError:
                print("Warning: 'relative_path' column not found. Attempting to load with all columns.")
                existing_df = pd.read_csv(output_csv)
                if 'relative_path' in existing_df.columns:
                    existing_df = existing_df[['relative_path']]
                else:
                    print("Warning: Could not identify relative_path column. Skipping deduplication.")
                    existing_df = pd.DataFrame()

            existing_records = set(existing_df['relative_path'].dropna().tolist()) if not existing_df.empty else set()
            print(f"Loaded {len(existing_records)} previously processed files to skip.\n")
        except Exception as e:
            print(f"Could not read existing CSV. Starting fresh. Error: {e}\n")

    print("Scanning folders and building task list...")

    tasks = []
    skipped_count = 0

    # Build the list of files we actually need to process
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                rel_path = os.path.relpath(dicom_path, source_folder)

                if rel_path in existing_records:
                    skipped_count += 1
                    continue

                png_filename = os.path.splitext(file)[0] + '.png'
                save_path = os.path.join(output_folder, png_filename)

                tasks.append((dicom_path, save_path, rel_path, file))

    # Optional sampling to limit the number of files processed in this run
    total_tasks = len(tasks)
    if NUMBER_TO_SAMPLE and total_tasks > NUMBER_TO_SAMPLE:
        random.seed(42)
        tasks = random.sample(tasks, NUMBER_TO_SAMPLE)
        print(f"Sampled {NUMBER_TO_SAMPLE} files from {total_tasks} available (deterministic).\n")

    if not tasks:
        print(f"\nNo new files to process. Skipped {skipped_count} existing files. Exiting.")
        return

    print(f"Found {len(tasks)} new files to process. Firing up CPU cores...\n")

    # --- Multiprocessing Execution ---
    cores_to_use = max(1, multiprocessing.cpu_count() - 1)

    metadata_list = []
    error_list = []
    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        for future in as_completed(futures):
            success, metadata, error_msg = future.result()

            if success and metadata is not None:
                metadata_list.append(metadata)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed and scraped {processed_count} / {len(tasks)} files...")
            else:
                failed_count += 1
                if error_msg:
                    error_list.append(error_msg)

    # --- Save Database ---
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        file_has_header = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
        df.to_csv(
            output_csv,
            mode='a' if file_has_header else 'w',
            header=not file_has_header,
            index=False
        )

    # --- Save Error Log ---
    if error_list:
        error_log_path = os.path.join(output_folder, "error_log.txt")
        with open(error_log_path, "a", encoding="utf-8") as f:
            for err in error_list:
                f.write(err + "\n")
        print(f"\n[!] Logged {len(error_list)} errors to {error_log_path}")

    print("\n" + "=" * 45)
    print(" TURBO PIPELINE COMPLETE")
    print(f" Successfully processed            : {processed_count} files")
    print(f" Skipped (already in database)    : {skipped_count} files")
    if failed_count > 0:
        print(f" Failed to process                : {failed_count} files")
    print("=" * 45)


if __name__ == "__main__":
    print_csv_headers()
    run_pipeline()
