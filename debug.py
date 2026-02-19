#!/usr/bin/env python3
import os

# Your exact path
folder_path = "/run/user/1000/gvfs/google-drive:host=gmail.com,user=novametrics.processing/0AASVi0RfSlEwUk9PVA/1hZtsxTVE-KIZxigFjNyBwBrVatA-PQ4p"

print(f"🔍 Checking: {folder_path}\n")

# Check if exists
print(f"Path exists: {os.path.exists(folder_path)}")
print(f"Is directory: {os.path.isdir(folder_path)}")

if os.path.exists(folder_path):
    try:
        # List all files
        all_files = os.listdir(folder_path)
        print(f"\n📁 Found {len(all_files)} total items:\n")
        
        for f in all_files:
            full_path = os.path.join(folder_path, f)
            is_dir = os.path.isdir(full_path)
            size = os.path.getsize(full_path) / (1024**3) if not is_dir else 0
            file_type = "[DIR]" if is_dir else f"[FILE {size:.2f}GB]"
            print(f"   {file_type} {f}")
        
        # Filter videos
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
        videos = [f for f in all_files if any(f.endswith(ext) for ext in video_extensions)]
        
        # Filter SRTs
        srt_extensions = ['.srt', '.SRT']
        srts = [f for f in all_files if any(f.endswith(ext) for ext in srt_extensions)]
        
        print(f"\n🎥 Video files: {len(videos)}")
        for v in videos:
            print(f"   - {v}")
        
        print(f"\n📝 SRT files: {len(srts)}")
        for s in srts:
            print(f"   - {s}")
            
    except Exception as e:
        print(f"\n❌ Error reading folder: {e}")
else:
    print("\n❌ Path does not exist!")
    print("\n💡 Try navigating in File Manager to find the correct path")