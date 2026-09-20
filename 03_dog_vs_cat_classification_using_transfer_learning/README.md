# Dog vs Cat Classification (Transfer Learning)

Classify photos as **dog** or **cat** by reusing a pretrained MobileNetV2 network (transfer learning).

## Dataset
- **Source:** Kaggle competition `dogs-vs-cats`
- **Size:** 25,000 labeled training photos (12,500 dogs, 12,500 cats). This project uses **2,000** of them (983 cats, 1,017 dogs).
- **Target:** 0 = cat, 1 = dog

## Approach
1. Downloaded and extracted the dataset with the Kaggle API.
2. Resized 2,000 images to 224 x 224 RGB and turned them into a NumPy array (2000, 224, 224, 3).
3. Created labels from the file names and split the data 80% / 20% (1,600 train, 400 test).
4. Scaled pixel values to 0-1.
5. Loaded **MobileNetV2** (feature vector from TensorFlow Hub, frozen) and added one Dense(2) output layer.
6. Trained with Adam and `SparseCategoricalCrossentropy(from_logits=True)` for 5 epochs.
7. Built a predictive system and tested it on two of my own photos (a cat and a dog, both predicted correctly).

## Results
| Metric | Value |
|--------|-------|
| Train accuracy (epoch 5) | 99.2% |
| **Test accuracy** | **98.0%** (392 of 400 correct) |
| Test loss | 0.041 |

## Limitations and next steps
- Only **2,000 of the 25,000** images are used. Using more data is an easy improvement.
- There is no validation set during training and no data augmentation (flips, rotations, zoom).
- The pretrained network is frozen. Fine-tuning the last layers could help.
- Only accuracy is reported. Add a confusion matrix and training curves.
- Good candidate to deploy as a small web app (upload a photo, get "dog" or "cat").

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. Create a free Kaggle account and an API token (Kaggle > Settings > Create New Token). Upload `kaggle.json` to your Colab session before running the download cell. Never commit `kaggle.json` to GitHub. Also accept the `dogs-vs-cats` competition rules on Kaggle, or the download fails.
3. Requires `tensorflow`, `tensorflow_hub` and `tf_keras`.
4. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
5. Run all cells.
