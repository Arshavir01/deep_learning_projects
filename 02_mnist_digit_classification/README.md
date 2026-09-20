# MNIST Digit Classification

Recognize handwritten digits (0 to 9) with a fully connected neural network.

## Dataset
- **Source:** MNIST, loaded with `keras.datasets.mnist` (downloads automatically)
- **Size:** 60,000 training images and 10,000 test images, 28 x 28 pixels, grayscale
- **Target:** the digit, 10 classes

## Approach
1. Checked shapes and labels, and displayed sample images.
2. Scaled pixel values from 0-255 to 0-1.
3. Built a neural network with Keras: Flatten, Dense(50, ReLU), Dense(50, ReLU), Dense(10, sigmoid).
4. Trained with Adam and `sparse_categorical_crossentropy` for 10 epochs.
5. Evaluated on the test set and plotted a **confusion matrix** heatmap.
6. Built a predictive system that takes your own digit image, converts it to grayscale, resizes it to 28 x 28 and predicts the digit.

## Results
| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 99.0% |
| **Test accuracy** | **96.9%** (96.85%) |
| Test loss | 0.128 |

## Limitations and next steps
- The gap between train accuracy (99.0%) and test accuracy (96.9%) shows some overfitting. There is no validation set during training.
- The output layer uses **sigmoid** with `sparse_categorical_crossentropy`. Use `softmax` so the 10 outputs form a probability distribution. In the notebook, one prediction gives about 1.0 for digit 7 and 0.95 for digit 9 at the same time.
- A fully connected network ignores the shape of the image. A CNN (as in the Fashion-MNIST project) usually does better on MNIST.
- The predictive system depends on the drawn image: MNIST digits are white on a black background, so photos or drawings may need inverting and cleaning.

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. MNIST downloads automatically. For the last section, upload your own handwritten digit image to `/content/MNIST_digit.png` (or change the path).
3. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
4. Run all cells.
