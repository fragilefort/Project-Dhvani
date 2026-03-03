# NNTI Project — Spoken Language Identification from Audio

**Model:** mHuBERT-147 fine-tuned with static/dynamic audio augmentation.  
**Dataset:** [badrex/nnti-dataset-full](https://huggingface.co/datasets/badrex/nnti-dataset-full)


## Python Version

**Python 3.10**

## Project Structure
Note: This project tree will still be updated after the report <br>
Note: The .env file will be provided empty, you will need to put HF and WANDB tokens
```
code/
├── train_model.py        # Task 1 & 2: fine-tuning with augmentation
├── analyze_model.py      # Task 3: confusion matrix, t-SNE, classification report
├── submit_job.sub        # HTCondor submission file for training
├── submit_analysis.sub   # HTCondor submission file for analysis
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
└── .env
```

## Environment Setup

### Docker and HTCondor on cluster (This is the only tested environment)

The project runs inside a Docker container based on `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`.

To run the first final state of the model:
```bash
cd code; condor_submit submit_job.sub
```
The results are expected to be in the same directory (code) after the job is finished.
To run the analysis of the model, choose the best checkpoint of the model, put the name of this checkpoint in the analyze_model.py (At the top), and also refer to this checkpoint in the submission file because it needs to be transferred into the container (and also needs to be zipped). Then:
```bash
cd code; condor_submit submit_analysis.sub
```

These are required to download the dataset from HuggingFace and log training metrics to Weights & Biases.

## Dependencies

Full list in `requirements.txt`:

```
torch==2.3.0
torchaudio==2.3.0
pandas==2.2.2
numpy==1.26.4
wandb==0.17.0
datasets==2.19.2
transformers==4.41.2
evaluate==0.4.2
huggingface_hub
python-dotenv
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
audiomentations==0.39.0
soundfile
accelerate>=0.21.0
```
