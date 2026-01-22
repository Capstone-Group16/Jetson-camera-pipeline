from huggingface_hub import hf_hub_download
import os

session = "20250828132109"

files = [
    f"{session}/D01_{session}.mp4",
    f"{session}/D02_{session}.mp4",
    f"{session}/D03_{session}.mp4",
    f"{session}/D04_{session}.mp4",
]

output_dir = "./excavator_session"   # <-- your folder in working dir
os.makedirs(output_dir, exist_ok=True)

for f in files:
    path = hf_hub_download(
        repo_id="FlywheelAI/excavator-dataset",
        filename=f,
        repo_type="dataset",
        local_dir=output_dir,              # <-- SAVE HERE
        local_dir_use_symlinks=False       # <-- COPY instead of symlink
    )
    print("Saved to:", path)
