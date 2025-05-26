import os
import shutil
from pathlib import Path
from git import Repo
from dotenv import load_dotenv
import subprocess

# Load the hf token
load_dotenv() 
token = os.getenv("HF_TOKEN")

folder_path = os.path.dirname(os.path.abspath(__file__))

def upload_files(repo_id, list, branch):
    # Temporary folder to clone the repo
    local_repo_dir = os.path.join(folder_path, "temp_hf_repo")
    if os.path.exists(local_repo_dir):
        shutil.rmtree(local_repo_dir)

    Repo.clone_from(
        f"https://:{token}@huggingface.co/{repo_id}.git",
        local_repo_dir,
        branch=branch
    )
    print('Repo cloned')
    
    # Initialize Git LFS
    subprocess.run(["git", "lfs", "install"], cwd=local_repo_dir)

    # Track .data-00000-of-00001 files with Git LFS
    subprocess.run(["git", "lfs", "track", "*.data-00000-of-00001"], cwd=local_repo_dir)
    print('Git LFS tracking set for *.data-00000-of-00001')

    for element in list:
        source_path = os.path.join(folder_path, element)
        target_path = os.path.join(local_repo_dir, element)

        # Delete the existing file or folder in the cloned repo, if it exists
        if os.path.exists(target_path):
            if os.path.isfile(target_path) or os.path.islink(target_path):
                os.remove(target_path)
            elif os.path.isdir(target_path):
                shutil.rmtree(target_path)

        # Copy the local file or folder into the cloned repo
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
        else:
            # Create the parent directories if they don't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    # Commit and push if there are changes
    repo = Repo(local_repo_dir)
    repo.git.add(all=True)
    if repo.is_dirty():
        repo.index.commit("Syncfolder from local to HF repo")
        origin = repo.remote(name='origin')
        origin.push(branch)
    else:
        print("No changes detected in folder.")

    shutil.rmtree(local_repo_dir)
    print("Temporary repo folder deleted.")

print('Hugging Face Space demo:')

space_demo_list = ["checkpoints", "demo", "Dockerfile",
              "config.py", "extract_spectrogram.py",
              "inference.py", "model.py", "requirements.txt"]

upload_files(repo_id = "spaces/LautaroOcho/Argentinian-audio-transcriptions-demo",
            list = space_demo_list,
            branch = 'main')

print('Hugging Face Space updated')

print('\nHugging Face Space app:')

space_app_list = ["data/df.csv", "checkpoints", "app.py",
              "config.py", "inference.py", "model.py", "requirements.txt"]

upload_files(repo_id = "spaces/LautaroOcho/Argentinian-audio-transcriptions-app",
            list = space_app_list,
            branch = 'main')

print('Hugging Face Space updated')

if __name__ == '__main__':
    from huggingface_hub import HfApi

    # Hugging Face Model
    print('\nHugging Face Model:')

    model_list = ["checkpoints"]

    upload_files(repo_id = "LautaroOcho/Argentinian-audio-transcriptions",
                list = model_list,
                branch = 'main')

    base_dir = Path(__file__).parent.resolve()
    folders = set()
    files = []

    for root, dirs, filenames in os.walk(base_dir):
        rel_root = Path(root).relative_to(base_dir)
        rel_root_str = rel_root.as_posix()

        # Add the current folder (if it's not the root)
        if rel_root_str != ".":
            folders.add(f"{rel_root_str}/**")

        # If we're inside a folder already marked to be ignored, skip its files and subdirectories
        if rel_root_str != "." and f"{rel_root_str}/**" in folders:
            dirs[:] = []  # prevent walking into subdirectories
            continue

        for filename in filenames:
            filepath = Path(root, filename).relative_to(base_dir).as_posix()
            files.append(filepath)

    # Combine and sort patterns
    ignore_patterns = list(folders.union(files))

    api = HfApi(token=token)
    # Hugging Face Dataset
    print('\nHugging Face Dataset:')
    not_ignore_in_dataset = ["data/**", "spectrogram/**"]

    api.upload_large_folder(
        folder_path=folder_path,
        repo_id="LautaroOcho/Argentinian-audio-transcriptions",
        repo_type="dataset",
        ignore_patterns=[elem for elem in ignore_patterns if elem not in not_ignore_in_dataset]
    )