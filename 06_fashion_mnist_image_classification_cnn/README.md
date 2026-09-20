# Fashion-MNIST Image Classification (CNN)

Classify grayscale photos of clothing into 10 categories with a Convolutional Neural Network.

## Dataset
- **Source:** Fashion-MNIST, loaded with `tensorflow.keras.datasets.fashion_mnist` (downloads automatically)
- **Size:** 60,000 training images and 10,000 test images, 28 x 28 pixels, grayscale
- **Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Approach
1. Set random seeds (0) for reproducible results.
2. Scaled pixel values to 0-1 and reshaped the images to (28, 28, 1).
3. Built a CNN with Keras: Conv2D(32), MaxPool, Conv2D(64), MaxPool, Conv2D(64), Flatten, Dense(64, ReLU), Dense(10).
4. Trained with Adam and `SparseCategoricalCrossentropy(from_logits=True)` for 5 epochs.
5. Plotted accuracy and loss curves and saved the trained model.

## Results
| Metric | Value |
|--------|-------|
| Train accuracy (epoch 5) | 92.1% |
| **Test accuracy** | **89.7%** |
| Test loss | 0.294 |

## Limitations and next steps
- The **test set was also used as the validation set** during training (`validation_data=(test_images, test_labels)`). Keep the test set for the very end: use `validation_split` or hold out part of the training data for validation.
- Train accuracy (92.1%) is above test accuracy (89.7%), so the model starts to overfit. Try Dropout, data augmentation and early stopping.
- Add a confusion matrix to see which classes get mixed up. Classes such as Shirt, T-shirt/top and Pullover are usually the hardest.
- The model is saved in the legacy `.h5` format. The newer `.keras` format is recommended.

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. Fashion-MNIST downloads automatically, no dataset file needed.
3. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
4. Run all cells.
