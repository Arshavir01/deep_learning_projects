# Face Mask Detection using CNN

Classify a face photo as **with mask** or **without mask** using a Convolutional Neural Network built from scratch.

> This project also has its own repository with more details: [DL_Project_FaceMaskDetection](https://github.com/Arshavir01/DL_Project_FaceMaskDetection).

## Dataset
- **Source:** Kaggle dataset [omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Size:** 7,553 images: 3,725 with mask and 3,828 without mask
- **Target:** 1 = with mask, 0 = without mask

## Approach
1. Downloaded and extracted the dataset with the Kaggle API.
2. Resized all images to 128 x 128 RGB and converted them to NumPy arrays (7553, 128, 128, 3).
3. Split the data 80% / 20% (6,042 train, 1,511 test) and scaled pixel values to 0-1.
4. Built a CNN with Keras: Conv2D(32) and MaxPool, Conv2D(64) and MaxPool, Flatten, Dense(128) with Dropout(0.5), Dense(64) with Dropout(0.5), and a Dense(2, sigmoid) output.
5. Trained with Adam and `sparse_categorical_crossentropy` for 5 epochs (20% of the training data used for validation).
6. Plotted accuracy and loss curves and built a predictive system for a single photo.

## Results
| Metric | Value |
|--------|-------|
| Train accuracy (epoch 5) | 92.3% |
| Validation accuracy (epoch 5) | 90.2% |
| **Test accuracy** | **89.5%** |
| Test loss | 0.330 |

## Limitations and next steps
- Validation accuracy was 92.7% in epoch 4 and dropped to 90.2% in epoch 5, so the training is not stable yet. Add data augmentation and early stopping.
- Try transfer learning (MobileNetV2, ResNet50 or EfficientNet). It usually gives a large gain on small image datasets.
- The output layer uses 2 sigmoid units with `sparse_categorical_crossentropy`. Use `softmax`, or 1 sigmoid unit with `binary_crossentropy`.
- Report precision, recall and a confusion matrix per class.
- Next step for deployment: convert the model to TensorFlow Lite and run it in an Android app or a real-time webcam demo.

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. Create a free Kaggle account and an API token (Kaggle > Settings > Create New Token). Upload `kaggle.json` to your Colab session before running the download cell. Never commit `kaggle.json` to GitHub.
3. The images are extracted to `/content/data/with_mask` and `/content/data/without_mask`. For the last section, upload a photo of a face and put its path in the prompt.
4. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
5. Run all cells.
