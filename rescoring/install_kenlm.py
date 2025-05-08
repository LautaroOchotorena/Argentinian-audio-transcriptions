import sys
import os
import subprocess
import shutil

# First install git-lfs
print('Installing git-lfs:\n')
subprocess.run(["sudo", "apt", "install", "git-lfs"], check=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(base_dir, "kenlm")):
    print('\nClone the repository needed:\n')
    # This will skip the lfs files
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    # Clone the repository
    subprocess.run(["git", "clone", "https://huggingface.co/edugp/kenlm"], check=True, env=env)

# Replace a file to work
print('\nReplacing a file to work better\n')
source =  os.path.join(base_dir, "ken_lm_model.py")
destination = os.path.join(base_dir, "kenlm")
# Remove the model file
old_path =  os.path.join(destination, "model.py")

if os.path.exists(old_path):
    os.remove(old_path)
# Copy and paste the new file
shutil.copy(source, destination)

os.chdir("kenlm")
# Initialize Git LFS (only necessary once, but it's okay to repeat)
print('Download the lfs needed for rescoring:\n')
subprocess.run(["git", "lfs", "install"], check=True)

# Corpus to use: wikepedia or oscar
include_paths = [
    "wikipedia/es.*",
    #"oscar/es.*"
]

# Convert to the correct format for the --include argument
include_arg = ",".join(include_paths)

# Run git lfs pull only for those files
subprocess.run(["git", "lfs", "pull", f"--include={include_arg}"], check=True)

if __name__ == '__main__':
    sys.path.append(os.path.abspath("kenlm"))
    from ken_lm_model import KenlmModel
    print('Loading KenLM model')
    # Load model trained on Spanish wikipedia
    model = KenlmModel.from_pretrained("wikipedia", "es", lower_case=True)
    print('Loaded')

    # Get perplexity
    print("Estoy orgulloso:", model.get_perplexity("Estoy orgulloso"))
    # Low perplexity, since sentence style is formal and with no grammar mistakes

    print("Teng dudas d todo:", model.get_perplexity("Teng dudas d todo"))
    # High perplexity, since the sentence contains grammar mistakes