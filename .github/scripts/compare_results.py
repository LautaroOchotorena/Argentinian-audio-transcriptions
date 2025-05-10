import os
from PIL import Image
import matplotlib.pyplot as plt

base_branch = os.environ.get("BASE_REF", "main")
head_branch = os.environ.get("HEAD_REF", "feature")

# Load images
img_main_loss = Image.open(f"metric_images/{base_branch}/loss_plot.png")
img_head_loss = Image.open(f"metric_images/{head_branch}/loss_plot.png")
img_main_wer = Image.open(f"metric_images/{base_branch}/wer_plot.png")
img_head_wer = Image.open(f"metric_images/{head_branch}/wer_plot.png")

# Create 2x2 figure
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Row 1: loss
axs[0, 0].imshow(img_main_loss)
axs[0, 0].set_title(f"{base_branch} - Loss")
axs[0, 0].axis('off')

axs[0, 1].imshow(img_head_loss)
axs[0, 1].set_title(f"{head_branch} - Loss")
axs[0, 1].axis('off')

# Row 2: WER
axs[1, 0].imshow(img_main_wer)
axs[1, 0].set_title(f"{base_branch} - WER")
axs[1, 0].axis('off')

axs[1, 1].imshow(img_head_wer)
axs[1, 1].set_title(f"{head_branch} - WER")
axs[1, 1].axis('off')

plt.tight_layout()
os.makedirs("comparison", exist_ok=True)
plt.savefig("comparison/comparison.png")