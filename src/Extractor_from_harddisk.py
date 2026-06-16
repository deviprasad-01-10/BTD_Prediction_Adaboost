import os
import shutil
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
MASTER_CSV = r"C:\Users\kalig\Downloads\embed_merged_without_blank_tissueden.csv"
PIPELINE_DB = r"C:\Users\kalig\OneDrive\Desktop\metadata_embed.csv"
HARD_DRIVE_BASE = r"D:\Datasets\EMBED" 
DESTINATION_BASE = r"C:\Sample Images_EMBED"

TARGET_PER_CLASS = 5000

def get_user_choice():
    """Interactive prompt to select which class to extract."""
    print("\n" + "="*50)
    print("EMBED DATASET EXTRACTOR - BATCH MANAGER")
    print("="*50)
    print("Which Tissue Density Class do you want to extract?")
    print("  [1] Class 1 (Fatty)")
    print("  [2] Class 2 (Scattered Fibroglandular)")
    print("  [3] Class 3 (Heterogeneously Dense)")
    print("  [4] Class 4 (Extremely Dense)")
    print("  [A] All Classes (5,000 each = 20,000 total)")
    print("-" * 50)
    
    while True:
        choice = input("Enter your choice (1, 2, 3, 4, or A): ").strip().upper()
        if choice in ['1', '2', '3', '4', 'A']:
            return choice
        print("Invalid input. Please enter 1, 2, 3, 4, or A.")

def extract_new_images():
    # Get the user's extraction goal before loading heavy data
    user_choice = get_user_choice()

    # ------------------------------------------
    # STEP 1: LOAD EXISTING IDs FROM V2.0 DATABASE
    # ------------------------------------------
    existing_ids = set()
    if os.path.exists(PIPELINE_DB):
        print(f"\nReading V2.0 Database: {PIPELINE_DB}...")
        try:
            db_df = pd.read_csv(PIPELINE_DB, usecols=['filename'])
            existing_ids = set(db_df['filename'].dropna().apply(lambda x: str(x).replace('.dcm', '')))
            print(f"Loaded {len(existing_ids)} previously processed images to exclude.")
        except Exception as e:
            print(f"Warning: Could not read pipeline DB. Proceeding cautiously. Error: {e}")
    else:
        print("\nNo existing Pipeline Database found. Starting completely fresh.")

    # ------------------------------------------
    # STEP 2: READ, FILTER, AND STRATIFY MASTER CSV
    # ------------------------------------------
    print(f"\nReading Master CSV file: {MASTER_CSV}...")
    try:
        df = pd.read_csv(MASTER_CSV, usecols=['anon_dicom_path', 'tissueden'])
    except Exception as e:
        print(f"CRITICAL: Failed to load Master CSV. Error: {e}")
        return

    df = df.dropna(subset=['anon_dicom_path', 'tissueden'])
    df['base_id'] = df['anon_dicom_path'].apply(lambda x: str(x).replace('\\', '/').split('/')[-1].replace('.dcm', ''))
    
    # Exclude files we already processed
    df_new_only = df[~df['base_id'].isin(existing_ids)]
    
    # Filter by user choice
    if user_choice != 'A':
        target_class = float(user_choice) # Ensure matching with pandas numeric types
        df_new_only = df_new_only[df_new_only['tissueden'] == target_class]
        print(f"\nFiltered dataset to ONLY show Class {user_choice}.")

    if len(df_new_only) == 0:
        print("No new files available for your selection. You may have processed all of them!")
        return

    print(f"\n--- SAMPLING OVERVIEW ---")
    sampled_dfs = []
    
    for tissue_class, group in df_new_only.groupby('tissueden'):
        available = len(group)
        n_sample = min(TARGET_PER_CLASS, available)
        print(f"Class {int(tissue_class)}: {available} available → Sampling {n_sample}")
        
        sampled_group = group.sample(n=n_sample, random_state=42)
        sampled_dfs.append(sampled_group)

    final_sampled_df = pd.concat(sampled_dfs)

    print(f"\nSuccessfully selected {len(final_sampled_df)} total files. Starting extraction...\n")

    # ------------------------------------------
    # STEP 3: EXTRACT AND ROUTE FILES BY CLASS
    # ------------------------------------------
    copied_count = 0
    skipped_count = 0
    missing_files = []

    # Iterating over rows so we have both the path AND the class label for folder routing
    for row in final_sampled_df.itertuples():
        relative_path = row.anon_dicom_path
        tissue_class = int(row.tissueden)
        
        clean_relative_path = os.path.normpath(relative_path.lstrip("\\/"))
        
        # Route the file into a class-specific folder (e.g., C:\Sample Images_EMBED\Class_2\...)
        class_folder = f"Class_{tissue_class}"
        
        source_file = os.path.normpath(os.path.abspath(os.path.join(HARD_DRIVE_BASE, clean_relative_path)))
        destination_file = os.path.normpath(os.path.abspath(os.path.join(DESTINATION_BASE, class_folder, clean_relative_path)))
        
        # Windows Long Path Bypass
        if os.name == 'nt':
            source_file = "\\\\?\\" + source_file
            destination_file = "\\\\?\\" + destination_file
        
        # Smart Resume
        if os.path.exists(destination_file):
            skipped_count += 1
            continue

        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, destination_file)
                copied_count += 1
                
                if copied_count % 500 == 0:
                    print(f"Extracted {copied_count} / {len(final_sampled_df)} files...")
            except Exception as e:
                print(f"Failed to copy {source_file}. Error: {e}")
        else:
            missing_files.append(source_file)

    # ------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------
    print("\n" + "="*40)
    print("EXTRACTION COMPLETE!")
    print(f"Skipped (already in folder): {skipped_count}")
    print(f"Successfully copied new    : {copied_count}")
    print(f"Total in Output Folders    : {skipped_count + copied_count} / {len(final_sampled_df)}")

    if missing_files:
        print(f"\nWARNING: Could not find {len(missing_files)} files on the D: drive.")

if __name__ == "__main__":
    try:
        extract_new_images()
    except Exception as e:
        print(f"Script terminated with an error: {e}")