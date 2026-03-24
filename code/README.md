
# NNTI Project - Spoken Language Identification from Audio

**Model:** mHuBERT-147 fine-tuned with static/dynamic audio augmentation.  
**Dataset:** [badrex/nnti-dataset-full](https://huggingface.co/datasets/badrex/nnti-dataset-full)


## Python Version

**Python 3.10**

## Project Structure
Note: The .env file will be provided empty, you will need to put HF and WANDB tokens
```
.
├── code
    ├── train_model.py        # Task 1 & 2: fine-tuning with augmentation
    ├── analyze_model.py      # Task 3: confusion matrix, t-SNE, classification report
    ├── submit_job.sub        # HTCondor submission file for training
    ├── submit_analysis.sub   # HTCondor submission file for analysis
    ├── requirements.txt      # Python dependencies
    ├── Dockerfile            # Docker image definition
    └── .env
    ├── README.md

├── report
    ├── acl_natbib.bst
    ├── custom.bib
    ├── eacl2023.sty
    ├── figures
    ├── main.pdf
    └── main.tex
```

## Environment Setup

### Docker and HTCondor on cluster (This is the only tested environment)

The project runs inside a Docker container based on `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`.

**The pre-built image is available at:**
```
fragilefort/project-dhvani:v4
```
No need to build the image - the submission files already reference this image. If you want to rebuild it yourself (e.g. after modifying `requirements.txt`):
```bash
docker build -t fragilefort/project-dhvani:v4 .
# then update the docker_image line in submit_job.sub and submit_analysis.sub
```

---

**Note: to view which experiments we did view the commit history using git log or visit https://github.com/fragilefort/Project-Dhvanio**

### Running Training
1. Update the `.env` file in the `code/` directory:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

2. Submit the training job:
```bash
cd code
condor_submit submit_job.sub
```

Results and checkpoints will be saved to `results/<run_name>/` in the same directory after the job finishes.

---

### Running Analysis

1. Identify the best checkpoint from the training results:
2. Zip the checkpoint folder:
```bash
cd results/<run_name>
zip -r checkpoint-XXXX.zip checkpoint-XXXX
```

3. Update `analyze_model.py` - set `CHECKPOINT` at the top of the file:
```python
CHECKPOINT = "checkpoint-XXXX"
```

4. Update `submit_analysis.sub` - point `transfer_input_files` to your zip:
```
transfer_input_files = .env, analyze_model.py, /full/path/to/checkpoint-XXXX.zip
```

5. Submit the analysis job:
```bash
cd code
condor_submit submit_analysis.sub
```

Output figures will be saved to `figures_<run_name>/` after the job finishes.
