# Deep Learning Projects

Six deep learning projects with TensorFlow / Keras, built while learning neural networks, CNNs and transfer learning. Each project has its own folder with a notebook and a README that explains the data, the approach, the results and what could be improved.

## Projects

| # | Project | Model | Test result |
|---|---------|-------|-------------|
| 1 | [Breast Cancer Classification](01_breast_cancer_classification/) | Dense neural network | 96.5% accuracy |
| 2 | [MNIST Digit Classification](02_mnist_digit_classification/) | Dense neural network | 96.9% accuracy |
| 3 | [Dog vs Cat Classification](03_dog_vs_cat_classification_using_transfer_learning/) | MobileNetV2 (transfer learning) | 98.0% accuracy |
| 4 | [CIFAR-10 Object Recognition](04_cifar_10_object_recognition_using_resnet50/) | ResNet50 (transfer learning) | 93.0% validation accuracy (test evaluation still to do) |
| 5 | [Face Mask Detection](05_face_mask_detection_using_cnn/) | CNN | 89.5% accuracy |
| 6 | [Fashion-MNIST Classification](06_fashion_mnist_image_classification_cnn/) | CNN | 89.7% accuracy |

> Face Mask Detection also has its own repository: [DL_Project_FaceMaskDetection](https://github.com/Arshavir01/DL_Project_FaceMaskDetection).

## A note on these results

These are single training runs. Some test sets are small (Breast Cancer has only 114 test samples), and the CIFAR-10 run was stopped before its last epochs, so only validation accuracy is reported there. Each project README lists its limitations and next steps.

## How to run

1. Clone the repo: `git clone https://github.com/Arshavir01/deep_learning_projects.git`
2. Open a project's `notebook.ipynb` in Google Colab. The notebooks use Colab helpers and `/content/` paths.
3. Projects 3, 4 and 5 download data from Kaggle and need a Kaggle API token (`kaggle.json`). Never commit this file to GitHub.
4. If you run locally, install the dependencies: `pip install -r requirements.txt`

## Tech

Python, TensorFlow, Keras, TensorFlow Hub, scikit-learn, OpenCV, NumPy, Matplotlib, Seaborn, Pillow

## Author

Arshavir Voskanyan
