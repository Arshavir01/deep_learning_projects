# Breast Cancer Classification

Classify a breast tumor as **malignant** or **benign** from 30 measurements, using a small neural network.

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, loaded with `sklearn.datasets.load_breast_cancer`
- **Size:** 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness and others)
- **Target:** 0 = malignant (212 samples), 1 = benign (357 samples)
- No missing values.

## Approach
1. Explored the data (shape, missing values, class balance, feature means per class).
2. Train/test split, 80% / 20% (455 train, 114 test).
3. Standardized the features with `StandardScaler`, fitted on the training data only.
4. Built a neural network with Keras: Flatten, Dense(20, ReLU), Dense(2, sigmoid).
5. Trained with Adam and `sparse_categorical_crossentropy` for 10 epochs (10% of the training data used for validation).
6. Plotted accuracy and loss curves, then built a predictive system for a single tumor.

## Results
| Metric | Value |
|--------|-------|
| Train accuracy (epoch 10) | 94.6% |
| Validation accuracy (epoch 10) | 95.7% |
| **Test accuracy** | **96.5%** (110 of 114 correct) |
| Test loss | 0.138 |

Always predicting "benign" would score about 63% (357 of 569), so the network learns real signal.

## Limitations and next steps
- The output layer uses **2 sigmoid units** with `sparse_categorical_crossentropy`. This loss expects probabilities that sum to 1, so use `softmax` (or 1 sigmoid unit with `binary_crossentropy`). The example prediction in the notebook, `[0.65, 0.46]`, shows the problem.
- The test set has only **114 samples** and the split is not stratified. Use cross-validation for a steadier estimate.
- For cancer detection, missing a malignant tumor is the costly mistake. Add a confusion matrix, recall and precision for the malignant class.
- Add a simple Logistic Regression baseline to show what the neural network adds.

## How to run
1. Open `notebook.ipynb` in Google Colab. These notebooks use Google Colab helpers (`cv2_imshow`) and file paths under `/content/`, so Google Colab is the easiest place to run them.
2. The dataset is built into scikit-learn, no download needed.
3. Install dependencies if needed: `pip install -r ../requirements.txt` (Colab already has most of them).
4. Run all cells.
