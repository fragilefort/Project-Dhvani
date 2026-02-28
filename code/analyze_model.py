import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import zipfile

load_dotenv()

print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

CHECKPOINT = "checkpoint-5400"

if os.path.exists(f"{CHECKPOINT}.zip"):
    print("Unzipping checkpoint...")
    with zipfile.ZipFile(f"{CHECKPOINT}.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Unzip complete.")

# using validation run from cluster
MODEL_PATH = CHECKPOINT
DATASET_NAME = "badrex/nnti-dataset-full"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(DATASET_NAME, split = "validation")

print(f"model loaded from {MODEL_PATH}")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
# output_hidden_states=True is required to get the representations for t-SNE
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH, output_hidden_states=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


label2id = model.config.label2id
id2label = model.config.id2label
lang_names = [id2label[i] for i in range(len(id2label))]

all_preds = []
all_labels_int = []
all_hidden_states = []
all_speakers = []

print("audios running through model")
with torch.no_grad():
    for i, item in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(dataset)}...")

        audio_array = item["audio_filepath"]["array"].astype("float32")

        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=-1).item()
        all_preds.append(pred)

        # convert string label to int using model's label2id
        all_labels_int.append(label2id[item["language"]])
        all_speakers.append(item["speaker_id"])

        # last hidden state mean pooled over time dimension
        hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
        all_hidden_states.append(hidden_state)

print("Confusion Matrix generated")

cm = confusion_matrix(all_labels_int, all_preds)

plt.figure(figsize=(18, 16))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=lang_names,
    yticklabels=lang_names
)



plt.title("Language Identification — Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Language", fontsize=12)
plt.ylabel("Actual Language", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()


report = classification_report(
    all_labels_int,
    all_preds,
    target_names=lang_names
)
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)


tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(np.array(all_hidden_states))

# plot by language
print("Generating t-SNE plot by language...")
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    hue=[id2label[l] for l in all_labels_int],
    palette="tab20",
    alpha=0.7,
    s=20
)
plt.title("t-SNE of Last Layer Representations (by Language)", fontsize=13)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_language.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_language.png"), dpi=150)
plt.close()
print("  t-SNE by language saved.")

# plot by speaker
print("Generating t-SNE plot by speaker...")
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    hue=all_speakers,
    palette="tab20",
    alpha=0.7,
    s=20,
    legend=False
)
plt.title("t-SNE of Last Layer Representations (by Speaker)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_speaker.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_speaker.png"), dpi=150)
plt.close()
print("  t-SNE by speaker saved.")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
