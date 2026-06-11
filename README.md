# Deep Learning for image reconstrcution from fMRI

## Introduction

The human visual cortex encodes visual stimuli and the brain's response to seen images can thus be analyzed thanks to fMRI. Deep learning methods to reconstruct those images from the corresponding neural activity is something than researchers have been trying to improve for several years now.

Recent methods implement complex and heavy architecture often including transformers and diffusion models to reconstruct images as faithful as possible to the original ones. 

For this project, I chose to implement a simpler architecture based on the method proposed by Beliy et al. in *From voxels to pixels and back* (2019). This method implements an encoder-decoder architecture relying on self-supervised learning to learn from small datasets.

The fundamental bottleneck is the scarcity of labeled (image, fMRI) pairs since acquiring paired data requires expensive scanner time. Self-supervision allows to exploit unlabeled data (images without fMRI, and test fMRI without images) to augment training and adapt the decoder to test-set statistics without additionnal acquisition cost.

## Objectives
- Implement a pipeline based on the Beliy et al. method:
    - retrieve the dataset
    - implement the Encoder-Decoder architecture
    - train on the dataset
- Evaluate results under different conditions
    - Compare results to Beliy et al.
    - Reduce the number of labeled data to measure changes on results quality and compare self-supervised to supervised-only methods
- Identify which regions of the visual cortex contributes most to reconstruction quality
    - select only a part of each fMRI data and measure changes in results


## Tools
Python: all data processing, modeling and analysis

Jupyter notebooks: reproducible pipelines with inline documentation

Git, GitHub: Version control, sharing

Google Colab: use GPU to train the models

Additionnal tools: PyTorch (deep learning), h5py (HDF5 reading), scikit-image, HuggingFace Transformers, nilearn (ROI brain visualization), matpolib (results display)


## Dataset

**Generic Object Decoding** (Horikawa & Kamitani, 2017)

Provides the image-fMRI pairs needed for supervised training and for evaluation. Publicly available on figshare (article ID 7387130). The raw BIDS-formatted data is also available on OpenNeuro (ds001246).

Five subjects participated in fMRI sessions while viewing images drawn from ImageNet.
The stimulus set consists of 1250 ImageNet images. 1200 images were used as training stimuli (one presentation per image per subject) and 50 images were held out as test stimuli (35 repeated presentations per image per subject).

Voxel responses are expressed as z-scores computed per voxel per run and restricted to the visual cortex (4466 voxels for Subject1)

Anatomical ROI labels are provided for V1, V2, V3, V4, LOC, FFA, PPA, LVC, HVC and VC.

This project uses data from Subject1 only. The 1200 training trials provide supervised (image, fMRI) pairs for encoder and decoder training. The 50 averaged test fMRI are used as test set and as unlabeled fMRI data.

Stimuli images are not directly provided with the dataset but can be asked with the form: https://forms.gle/ujvA34948Xg49jdn9

## Deliverables

- **fmri_to_image_pipeline.ipynb** : Notebook with whole pipeline step by step
   - Step 1 : dataset downloading and creation of training and testing pairs
   - Step 2 : Encoder + Decoder implementation and training
   - Step 3 : Evaluation
- **Experiment_ROI.ipynb**: Notebook using the pipeline to train and compare models on different parts of the visual cortex
- **experiment_dataset_size.ipynb**: Notebook using the pipeline to train and compare supervised-only vs self-supervised methods on different dataset sizes


- Results: image reconstructions, metrics and comparison in the results/ folder (how each results was obtained and by which model is decribed in the **Results** part of this report)

- The trained models are not direclty provided due to their size (>200 Mo) but explanation on how they were obtained is porvided

## Architecture and Methods

### Encoder E: image to fMRI prediction

```
Input: RGB image (3 x 112 x 112)
    AlexNet conv1 (ImageNet pretrained, frozen)
    BatchNorm2d(64)
    Conv3x3 stride=2, 32 channels, BN, ReLU  (x2)
    FC to fMRI prediction (4466,)
```

### Decoder D: fMRI to image reconstruction

```
Input: fMRI vector (4466,)
    FC, reshape (64 x 14 x 14)
    Conv3x3 + ReLU + Upsample×2 + BN  (x3)
    Conv3x3 + Sigmoid to RGB image (3 x 112 x 112)
```

### Training : Phase 1: Encoder (supervised)

SGD with momentum, CosineAnnealingLR, 80 epochs.

Loss: `L_r = α × MSE(E(img), fmri) − (1−α) × cosine(E(img), fmri)` with α=0.9

### Training : Phase 2: Decoder (self-supervised)

Adam, StepLR (×0.2 every 30 epochs), 120-150 epochs. Encoder frozen.

Three loss components:
- **L_D** (supervised): `image_loss(D(fmri), img)` on labeled pairs
- **L_ED** (unlabeled images): `image_loss(D(E(img_unlab)), img_unlab)` : forces D-E to be an identity on natural images
- **L_DE** (test fMRI without images): `fmri_loss(E(D(fmri_test)), fmri_test)` : adapts the decoder to test fMRI statistics

`total = lambda_d * L_D + lambda_ed * L_ED + lambda_de * L_DE`

lambdas coefficients can be adapted depending on the desired model

Image loss: `L1 + 0.1 * VGG19_features + 0.001 * TV`

### Evaluation metrics

- **PixCorr**: Pearson correlation between reconstruction and original pixels
- **SSIM**: Structural Similarity Index (scikit-image)
- **CLIP similarity**: cosine similarity between CLIP ViT-B/32 embeddings (captures high level features)
- **N-way identification accuracy**: for each reconstruction, identify the correct original image among n candidates by pixel correlation (n=2, 5, 10)

## Results

### Results folder structure

```
results/
    reconstructions/                All 50 test image reconstructions per model
    nway_accuracy/                  N-way identification accuracy plots 
    snr_curves/                     Accuracy vs number of repetitions
    other_metrics/                  PixCorr, SSIM, CLIP similarity histograms
    roi_experiment/                 ROI comparison plots
    data_size_experiment/           Supervised vs self-supervised comparison
```

### Main pipeline
The results of three models trained on the main pipeline are available in the results/ folder:
- **model_full_self_supervised**: (trained with lambda_d = 1, lambda_de = 1, lambda_ed = 1)
- **model_image_self_supervised**: (trained with lambda_d = 1, lambda_ed = 1, lambda_de = 0)
- **model_only_supervised**: (trained with lambda_d = 1, lambda_ed = 0, lambda_de = 0)

The following histrograms (available at results/nway_accuracy/{model_name}_decoder_nway_accuracy.png) show the 2-way, 5-way, 10-way accuracy of the three models compared to the results reported in the Beliy et al. paper and to chance.

![N-way accuracy](results/nway_accuracy/model_full_self_supervised_decoder_nway.png)

![N-way accuracy](results/nway_accuracy/model_image_self_supervised_decoder_nway.png)

![N-way accuracy](results/nway_accuracy/model_only_supervised_decoder_nway.png)

The obtained results are significantly above chance, which shows the ability of the architecture to reconstuct recognisable features, though it does not reach the Beliy et al. performances.

Reconstructions

SNR

Losses

which is why, for all following models, the coefficients for self-supervised models will be (lambda_d = 1, lambda_ed = 1, lambda_de = 0.05)