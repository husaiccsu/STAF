## STAF: A Sequence–Topology Alignment and Fusion Framework with Protein–GO Contrastive Learning and Gradient Boosting for Protein Function Prediction

STAF is a computational framework designed for large-scale automated protein function prediction. It natively integrates sequence information, protein-protein interaction (PPI) network topologies, and hierarchical Gene Ontology (GO) terms using a dual-branch graph encoder, contrastive learning, and an XGBoost-based auxiliary supervision mechanism. 

## Prerequisites

*   **Python Version:** Python 3.10 is required.
*   **Hardware:** An NVIDIA GPU (e.g., RTX 2080Ti) is highly recommended to accelerate deep learning model training. 
*   **Environment Setup:** Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The script expects specific datasets and pre-trained models to be placed in dedicated local directories. Ensure your directory structure mirrors the following layout:

*   **CAFA3 Annotations:** Place `cafa3_eukaryon.txt` in the `./data/` directory.
*   **PPI Network:** Place the high-confidence STRING v12 network file as `PPI_StringV12_min700.txt` in the `./data/` directory.
*   **Protein Sequences:** Place the UniProt FASTA file as `uniprot_Sequence.fasta` in the `./data/` directory.
*   **Pre-trained ESM Models:** The framework requires ESM-1b and ESM-2 (3B) models for feature extraction. You can download these models from Hugging Face (e.g., `facebook/esm1b_t33_650M_UR50S` and `facebook/esm2_t36_3B_UR50D`). Once downloaded, save them in the `./local_models/` directory so that the model paths resolve to `./local_models/esm1b_t33_650M_UR50S` and `./local_models/esm2_t36_3B_UR50D`.

## Feature Extraction

Before training the main STAF model, you must generate the protein sequence features using the pre-trained ESM language models.

1.  Open `STAF.py` and uncomment the feature extraction functions `extract_ppi_protein_features("esm1b")` and `extract_ppi_protein_features("esm2_3b")` within the data loading sections if you are running this for the first time.
2.  The script will automatically tokenize the sequences, generate residue-level representations, and save the pooled features as `.npz` files in the `./data/` directory to speed up future runs.

## Training and Evaluation

The core script is configured to perform a standard 10-fold cross-validation over the dataset. 

1.  Execute the main Python script to start the pipeline:
    ```bash
    python STAF.py
    ```
2.  The framework will automatically split the nodes into ten mutually exclusive subsets to prevent information leakage.
3.  The model optimizes a joint objective combining Asymmetric Loss (ASL) for hard label supervision, XGBoost-generated probabilities for soft supervision, and a multi-positive InfoNCE loss for contrastive learning.
4.  Evaluation metrics, including Fmax, AUROC, and AUPR, will be printed to the console for each fold and aggregated at the end of the run.
