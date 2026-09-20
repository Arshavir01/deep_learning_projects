# CIFAR-10 Object Recognition (ResNet50)

Recognize 10 kinds of objects in small color images using a pretrained ResNet50 network.

## Dataset
- **Source:** Kaggle competition `cifar-10` (labeled training set with `trainLabels.csv`)
- **Size:** 50,000 labeled images, 32 x 32 pixels, RGB
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Approach
1. Downloaded the data with the Kaggle API, extracted `train.7z` and read the labels from `trainLabels.csv`.
2. Loaded all images into a NumPy array and split the data 80% / 20% (40,000 train, 10,000 held out).
3. Scaled pixel values to 0-1.
4. Defined a small fully connected baseline (Flatten, Dense(64), Dense(10)). Its results were not saved in the notebook.
5. Built the main model on **ResNet50** with ImageNet weights: three 2x upsampling layers (32 x 32 to 256 x 256), the ResNet50 base, then Flatten, BatchNormalization, Dense(128), Dropout(0.5), BatchNormalization, Dense(64), Dropout(0.5), BatchNormalization and Dense(10, softmax).
6. Trained with RMSprop (learning rate 2e-5), 10% of the training data used for validation.

## Results
The saved training run covers **4 complete epochs** and was stopped during epoch 5 (10 were planned).

| Epoch | Train accuracy | Validation accuracy |
|-------|----------------|---------------------|
| 1 | 32.2% | 76.3% |
| 2 | 68.5% | 89.3% |
| 3 | 81.4% | 92.1% |
| 4 | 87.2% | **93.0%** |

**No test-set result yet:** the notebook has no `model.evaluate` call on the held-out 10,000 images.

## Limitations and next steps
- Re-run all 10 epochs, then evaluate on the held-out images and report the **test accuracy**. This is the most important missing piece.
- Save the baseline's results too, so you can show how much ResNet50 improves on a simple network.
- Training is slow (about 7 minutes per epoch on Colab) because images are upscaled 8 times. Try a smaller input size or a lighter backbone.
- Add data augmentation, early stopping, a confusion matrix and per-class accuracy.

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. Create a free Kaggle account and an API token (Kaggle > Settings > Create New Token). Upload `kaggle.json` to your Colab session before running the download cell. Never commit `kaggle.json` to GitHub. Also accept the `cifar-10` competition rules on Kaggle, or the download fails.
3. Use a GPU runtime (Runtime > Change runtime type). Training takes a long time on CPU.
4. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
5. Run all cells.
