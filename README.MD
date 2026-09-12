# Solar Filament Segmentation with U-Net

A U-Net-based image segmentation pipeline developed for the **Solar Filament Segmentation Challenge 2026** using H-alpha solar observations from the MAGFiLO dataset.

The goal of the competition is to generate accurate segmentation masks for individual solar filaments while preserving fine-scale filament structures and minimizing background regions. The final evaluation emphasizes **Panoptic Quality (PQ)**, together with Dice, IoU, one-to-many/many-to-one relationships, qualitative morphology, and code quality.

## Overview

Solar filaments are dense structures of solar material suspended above the photosphere. Their morphology and evolution are important for space-weather research because filaments are associated with solar eruptions such as coronal mass ejections and solar flares.

This project uses a **U-Net semantic segmentation model** to generate pixel-level filament probability maps. Post-processing then converts the binary segmentation into individual filament instances for Panoptic Quality evaluation and competition submission.

## Pipeline

The complete notebook implements the following pipeline:

1. Load H-alpha training and test images.
2. Load the COCO-style MAGFiLO annotations.
3. Explore filament counts and annotation areas.
4. Convert polygon annotations into binary masks.
5. Split the training images into training and validation sets.
6. Resize images and masks to **512 × 512**.
7. Normalize H-alpha images to the **[0, 1]** range.
8. Train a U-Net segmentation model.
9. Generate validation probability maps.
10. Optimize the probability threshold and minimum component area.
11. Extract individual filament instances using connected components.
12. Match predicted and ground-truth instances using IoU-based one-to-one matching.
13. Calculate Panoptic Quality, Dice, and IoU.
14. Analyze fragmentation, merging, false positives, and false negatives.
15. Visualize representative segmentation cases.
16. Retrain the final U-Net using the complete training set.
17. Perform inference on the provided test H-alpha images.
18. Extract individual test filament instances and encode them using COCO RLE.
19. Validate and save the final `submission.csv`.

## Model Architecture

The segmentation model is a U-Net designed for single-channel H-alpha images.

### Encoder

The encoder uses four convolutional blocks:

- `1 → 32`
- `32 → 64`
- `64 → 128`
- `128 → 256`

Each block contains two `3 × 3` convolutional layers with Batch Normalization and ReLU activation, followed by max pooling between encoder stages.

### Bottleneck

The bottleneck uses:

- `256 → 512`

with the same double-convolution structure.

### Decoder

The decoder uses transposed convolutions for upsampling and skip connections from the corresponding encoder stages:

- `512 → 256`
- `256 → 128`
- `128 → 64`
- `64 → 32`

The final `1 × 1` convolution produces a single-channel filament logit map.

## Training Configuration

| Parameter | Value |
|---|---:|
| Input channels | 1 |
| Image size | 512 × 512 |
| Batch size | 8 |
| Epochs | 15 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss + Dice Loss |
| Positive-class weight | 10.0 |
| Random seed | 42 |

The positive-class weighting is used together with Dice loss to address the highly imbalanced pixel distribution between filament and background regions.

## Instance Segmentation and Post-processing

The U-Net produces a probability map rather than directly predicting separate filament instances.

The probability map is converted into a binary mask using an optimized threshold. Small components below the optimized minimum-area threshold are removed. The remaining connected components are treated as individual filament instances.

For the final validation run recorded in the notebook, the selected configuration was:

- **Instance method:** Connected Components
- **Probability threshold:** 0.800
- **Minimum component area:** 50 pixels
- **PQ IoU matching threshold:** 0.50

## Evaluation

The competition's primary quantitative metric is **Panoptic Quality (PQ)**. The notebook therefore evaluates the predictions at the instance level rather than relying only on pixel-level segmentation metrics.

The validation pipeline includes:

- Panoptic Quality
- Matched Dice
- Matched IoU
- True positives
- False positives
- False negatives
- Dice distribution
- IoU distribution
- One-to-many relationships
- Many-to-one relationships
- Error-case visualization

### Final Validation Result

The final validation run in the notebook produced:

| Metric | Result |
|---|---:|
| Panoptic Quality | **0.2525** |
| Mean matched Dice | **0.7544** |
| Mean matched IoU | **0.6089** |

The validation results are based on the validation split used by the notebook and should not be interpreted as the hidden test-set leaderboard score.

## Qualitative Evaluation

Representative validation cases are visualized using:

- Original H-alpha image
- Ground-truth segmentation
- Binary prediction
- Predicted filament instances
- Prediction overlay

The notebook specifically examines:

- Worst-PQ validation case
- Best-PQ validation case
- Most false-positive case
- Most false-negative case

These visualizations provide a qualitative assessment of whether the predicted masks follow the apparent morphology of solar filaments and whether individual structures are fragmented or incorrectly merged.

## Competition Compliance

The pipeline is designed around the competition's stated evaluation requirements.

For test inference, the model uses **only the H-alpha images provided in the test directory**. The training process uses the provided training annotations to construct segmentation masks; no additional ground-truth metadata is required for inference.

The final submission contains one row for each predicted filament instance with:

```text
filament_id
segmentation_rle
```

RLE masks are encoded using `pycocotools` and represent the original image dimensions.

## Final Submission

The notebook generated a submission containing:

- **180 test images**
- **1,199 predicted filament instances**
- **1,199 unique filament IDs**

The generated submission is validated for:

- Correct column names
- Unique filament IDs
- Non-empty RLE values
- Test-image ID consistency
- Successful RLE decoding
- Correct original mask dimensions

## Reproducibility

The complete pipeline is contained in the Jupyter notebook:

```text
notebook/
└── solar-filament-segmentation-challenge-2026-unet.ipynb
```

The notebook also saves the following outputs during execution:

```text
filament_unet_final.pth
submission.csv
final_configuration.json
final_validation_results.csv
```

The final configuration records the random seed, image size, batch size, number of epochs, learning rate, loss, positive-class weight, optimized threshold, minimum component area, instance extraction method, and PQ IoU threshold.

## Repository Structure

```text
solar-filament-segmentation/
│
├── notebook/
│   └── solar-filament-segmentation-challenge-2026-unet.ipynb
│
├── output/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

The main Python dependencies used by the notebook are listed in `requirements.txt`.

The development environment used for the notebook was:

- Python 3.12.13
- NumPy 2.0.2
- pandas 2.3.3
- Matplotlib 3.10.0
- Pillow 11.3.0
- OpenCV 4.13.0
- PyTorch 2.10.0
- scikit-learn 1.6.1
- SciPy 1.16.3
- pycocotools 2.0.11

## Running the Notebook

The notebook was developed and executed in **Kaggle Notebook**.

To reproduce the competition pipeline:

1. Open the notebook in Kaggle.
2. Attach the competition dataset.
3. Install/use the packages specified in `requirements.txt`.
4. Run the notebook from the first cell through the final cell.
5. The final submission will be generated as:

```text
/kaggle/working/submission.csv
```

The notebook performs training, validation, instance extraction, test inference, RLE encoding, submission validation, and final configuration saving as part of the complete pipeline.

## Acknowledgment

This project uses the **MAGFiLO (Manually Annotated GONG Filaments from H-Alpha Observations)** dataset as the ground-truth source for solar filament segmentation.

The competition focuses on segmentation masks of solar filaments and evaluates submissions primarily using Panoptic Quality, together with additional quantitative and qualitative criteria.
