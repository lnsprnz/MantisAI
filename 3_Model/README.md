# Model

## **[Baseline Model & Model Training](mantis_keypoint_detection_training)**

## **Overview**
This project trains a deep learning model for **keypoint detection on mantises** using **convolutional neural networks (CNNs)**. Two models are implemented:
1. **Baseline Model** – A simple CNN trained from scratch.
2. **Trained Model** – Uses **transfer learning** (e.g., MobileNetV2) for improved performance.

## **Installation**
Install the required dependencies:

```sh
pip install imgaug keras tensorflow scikit-learn pandas numpy
```

## **Dataset**
The dataset consists of:
- **Mantis images** (`train/`)
- **JSON annotations** defining keypoints (`MantisTrain.json`)
- **CSV keypoint definitions** (`MantisKeypointDef.csv`)

Update the paths accordingly:

```python
IMG_DIR = "path/to/train/images"
JSON = "path/to/MantisTrain.json"
KEYPOINT_DEF = "path/to/MantisKeypointDef.csv"
```

## **Baseline Model**
A simple CNN trained from scratch:

```python
baseline_model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_KEYPOINTS)
])
```

## **Trained Model (Transfer Learning)**
Uses **MobileNetV2** as a feature extractor:

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_KEYPOINTS)
])
```

## **Training the Models**
Run the notebook to train both models:

1. Train the **baseline model**.
2. Train the **transfer learning model**.
3. Compare the performance with **[Prediction](mantis_keypoint_detection_prediction)**


## **[Prediction](mantis_keypoint_detection_prediction)**
**[Evaluation OKS](evaluation)**
**[Evaluation Boxplot](evaluation_boxplot)**
