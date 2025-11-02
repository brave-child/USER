# USER: User-Side Modality Representation Enhancement for Multimodal Recommendation


## Dataset

We provide three processed datasets: Baby, Sports, Clothing.

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1BxObpWApHbGx9jCQGc8z52cV3t9_NE0f?usp=sharing)

## Training
  ```
  python src/main.py
  ```


## Data Preparation
You should first get the textual token embedding by running save_token_embeddings.py with transformers library (BERT, RoBERTa, LlaMA). You can first try USER on the pre-processed datasets baby, sports, and clothing. We provide VQGAN / BEiT tokens and codebook for visual modality and BERT / RoBERTa / LlaMA tokens and codebook for textual modality. 
Download from Google Drive: [Tokens and Codebook](https://drive.google.com/drive/folders/1IXLbNNzyPMiuOUyZPzH1s-J-VAuULYBT?usp=drive_link)

rename `.json` file as `tokens.json`, 

rename `.pth` file as `codebook.json`, 

Place the `tokens.json` file into the corresponding dataset folder (e.g., `data/baby`), 

Place the `codebook.pth` file into the `codebook` directory.


## Other Modality
You can explore the generalization ability of our method across different modalities by replacing the visual or textual modality in the code with features from other modalities.

For example, using the [MicroLens](https://drive.google.com/drive/folders/18MjrDgfUh-er6dQnHTTEPHDbe-djKr3p) dataset, we provide its codebook and tokens, which include textual, visual, and video modalities.
