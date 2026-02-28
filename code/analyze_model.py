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

load_dotenv()

# using validation run from cluster
MODEL_PATH = "results_142014"
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

all_preds = []
all_labels = []
all_hidden_states = []
all_speakers = []

print("audios running through model")
with torch.no_grad():
    for item in dataset:
        audio_array = item["audio"]["array"]
        
        inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # prediction for Confusion Matrix
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        all_preds.append(pred)
        
        all_labels.append(item["language"]) 
        all_speakers.append(item["speaker_id"]) 

        # Hidden States for t-SNE
        hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
        all_hidden_states.append(hidden_state)

print("Confusion Matrix generated")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues") 
plt.title("Language Confusion Matrix")
plt.xlabel("Predicted Language")
plt.ylabel("Actual Language")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.pdf"))
plt.close()

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(np.array(all_hidden_states))

print("t-SNE plot by Language")
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=all_labels, palette="tab20", legend=False)
plt.title("t-SNE of Last Layer (Colored by Language)")
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_language.pdf"))
plt.close()

print("t-SNE plot by Speaker Identity...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=all_speakers, palette="viridis", legend=False)
plt.title("t-SNE of Last Layer (Colored by Speaker)")
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_speaker.pdf"))
plt.close()

print(f"plots saved to {OUTPUT_DIR}")
