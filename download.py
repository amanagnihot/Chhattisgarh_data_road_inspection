import gdown
import os
import time

# --- YOUR LINKS ---
FOLDER_LINKS = [
    "https://drive.google.com/drive/folders/1BopTWABa4V6I4SraRaSbY1yU3mVzlYSx?usp=sharing",
    "https://drive.google.com/drive/folders/1OTM1QMwBaA9C-xqpRhmzA1VlQsmNkl3p?usp=sharing",
    "https://drive.google.com/drive/folders/17hcVdOtu9369y5C34TXaKwvHVubPn7f0?usp=sharing"
    ]

# ------------------

def download_mp4_only():
    for f_index, folder_url in enumerate(FOLDER_LINKS, start=1):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Checking Folder {f_index} for MP4 files...")

        try:
            # 1. Get the list of files in the folder without downloading them yet
            files = gdown.list_objects(folder_url, fuzzy=True)
            
            # 2. Filter the list for .mp4 files
            mp4_files = [f for f in files if f.name.lower().endswith('.mp4')]
            
            if not mp4_files:
                print(f"No MP4 files found in folder {f_index}. Skipping...")
                time.sleep(2)
                continue

            print(f"Found {len(mp4_files)} videos. Starting serial download...\n")

            # 3. Download each MP4 file one by one
            for v_index, file_obj in enumerate(mp4_files, start=1):
                print(f"[{v_index}/{len(mp4_files)}] Downloading: {file_obj.name}")
                
                # Construct the direct download URL for the specific file
                file_url = f"https://drive.google.com/uc?id={file_obj.id}"
                
                # Download the file
                gdown.download(file_url, output=file_obj.name, quiet=False, fuzzy=True)
                print(f"Done: {file_obj.name}\n")

        except Exception as e:
            print(f"Error processing folder {f_index}: {e}")
            time.sleep(5)

    print("\n🎉 All MP4 downloads finished!")

if __name__ == "__main__":
    download_mp4_only()