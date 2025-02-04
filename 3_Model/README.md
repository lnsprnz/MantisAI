# Model

# **[Baseline Model & Model Training](mantis_keypoint_detection_training)**

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


# **[Prediction](mantis_keypoint_detection_prediction)**

This project uses a **trained deep learning model** to predict **mantis keypoints** in images. The model was trained using **transfer learning** and can now detect **five keypoints** on new test images.

## **Requirements**
Before running the prediction script, install the required dependencies:

```sh
pip install keras tensorflow imgaug scikit-learn pandas numpy matplotlib pillow
```

Ensure that you have a trained model available from the previous training notebook **[Baseline Model & Model Training](mantis_keypoint_detection_training)**.

## **Dataset**
The test dataset should include:
- **Test images directory** (`test/`)
- **JSON annotation file** (`MantisTest.json`)
- **Keypoint definition CSV** (`MantisKeypointDef.csv`)

Update the paths accordingly:

```python
IMG_DIR_TEST = "path/to/test/images"
JSON_TEST = "path/to/MantisTest.json"
KEYPOINT_DEF = "path/to/MantisKeypointDef.csv"
```

## **Loading the Model**
Make sure to load the trained model from  before making predictions:

```python
from tensorflow.keras.models import load_model

model = load_model("path/to/trained_model.h5")
```

## **Performing Predictions**
To make predictions:
1. Load the test images and their corresponding JSON annotations.
2. Preprocess the images (resize, normalize).
3. Run the model to predict keypoints.
4. Visualize the predicted keypoints.

Example code for predictions:

```python
def predict_keypoints(model, image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    return predictions.reshape(-1, 2)  # Convert flat output into (x, y) pairs
```

## **Visualizing Predictions**
To visualize the predicted keypoints:

```python
import matplotlib.pyplot as plt

def plot_predictions(image_path, keypoints):
    img = Image.open(image_path)
    plt.imshow(img)
    
    for (x, y) in keypoints:
        plt.scatter(x * img.width, y * img.height, c='red', marker='o')
    
    plt.show()

# Example usage
image_path = "path/to/test/image.jpg"
predicted_keypoints = predict_keypoints(model, image_path)
plot_predictions(image_path, predicted_keypoints)
```

## **Expected Results**
- The model should detect and **plot five keypoints** on the test image.
- The performance may vary depending on training quality and dataset diversity.

# **[Evaluation OKS](evaluation)**

This project evaluates the performance of a trained **mantis keypoint detection model** using **Object Keypoint Similarity (OKS)**. The OKS metric quantifies the accuracy of keypoint predictions by comparing them to ground-truth annotations.

## **What is OKS?**
- OKS measures the **similarity** between predicted and ground-truth keypoints.
- It considers **spatial distance**, **object scale**, and **keypoint visibility**.
- **OKS values range from 0 to 1**, where **1 indicates a perfect match**.
- A **threshold of 0.75** is used:  
  - OKS ≥ 0.75 → **True Positive**  
  - OKS < 0.75 → **False Positive**

## **Installation**
Install the required dependencies:

```sh
pip install torch numpy pandas matplotlib
```

## **Evaluation Process**
1. **Load the test dataset** (keypoints and areas).
2. **Compute OKS** between predicted and ground-truth keypoints.
3. **Determine accuracy** based on the 0.75 threshold.

### **Loading Data**
Ensure that the test dataset paths are correctly set:

```python
IMG_DIR_TEST = "path/to/test/images"
JSON_TEST = "path/to/MantisTest.json"
KEYPOINT_DEF = "path/to/MantisKeypointDef.csv"
```

### **Running the OKS Evaluation**
OKS is computed as follows:

```python
import torch

def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    """
    Calculate Object Keypoint Similarity (OKS).

    Params:
        gt_kpts: Ground-truth keypoints [M, #kpts, 3] (x, y, visibility)
        pred_kpts: Predicted keypoints [N, #kpts, 3]
        sigmas: Standard deviations for each keypoint type
        areas: Areas of ground-truth objects [M,]

    Returns:
        oks: OKS scores [M, N]
    """
    EPSILON = torch.finfo(torch.float32).eps  # Avoid division by zero
    dist_sq = (gt_kpts[:, None, :, 0] - pred_kpts[..., 0])**2 + (gt_kpts[:, None, :, 1] - pred_kpts[..., 1])**2
    vis_mask = gt_kpts[..., 2].int() > 0  # Check visibility
    k = 2 * sigmas
    denom = 2 * (k**2) * (areas[:, None, None] + EPSILON)
    exp_term = dist_sq / denom
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)
    return oks
```

### **Interpreting the Results**
- **OKS scores** are calculated for each keypoint.
- **Final model accuracy** is based on how many OKS values exceed 0.75.

```python
threshold = 0.75
true_positives = (oks_scores >= threshold).sum()
false_positives = (oks_scores < threshold).sum()

accuracy = true_positives / (true_positives + false_positives)
print(f"Model Accuracy: {accuracy:.2%}")
```

## **Expected Output**
- A percentage score indicating the **model's accuracy** in detecting keypoints.

# **[Evaluation Boxplot](evaluation_boxplot)**


This project evaluates the performance of a **mantis keypoint detection model** by visualizing the **Euclidean distances** between predicted keypoints and ground-truth keypoints using **boxplots**. The goal is to analyze the **error distribution** rather than relying on Object Keypoint Similarity (OKS), which performed poorly due to model inaccuracies.

## **Why Use Boxplots?**
- Boxplots provide a **clear visualization** of prediction errors.
- They show the **distribution of keypoint errors** across different keypoints.
- Help identify **outliers and overall model accuracy trends**.

## **Installation**
Install the required dependencies:

```sh
pip install numpy matplotlib json
```

## **Data Preparation**
The input data consists of a **JSON file** containing **predicted** and **ground-truth keypoints**.

- The JSON file should follow this format:

```json
[
    {
        "predictions": [[x1, y1], [x2, y2], ..., [x5, y5]],
        "ground_truth": [[x1, y1], [x2, y2], ..., [x5, y5]]
    },
    ...
]
```

- **Update the file path** in the script:

```python
json_path = "path/to/predictions.json"
```

## **Distance Calculation**
The script computes the **Euclidean distance** between each predicted and ground-truth keypoint:

```python
import numpy as np

def calculate_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
```

## **Boxplot Visualization**
To visualize keypoint prediction errors:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot(boxplot_data, labels=[f'Keypoint {i+1}' for i in range(5)])
plt.title('Distances Between Ground Truth and Predictions for Each Keypoint')
plt.ylabel('Distance (pixels)')
plt.grid(axis='y')
plt.show()
```

## **Interpreting the Boxplots**
- **Small interquartile range (IQR)** → Low variance (better accuracy).
- **Large spread or many outliers** → Model struggles with specific keypoints.
- **Keypoints with consistently high errors** → Likely **need more training data** or **better model fine-tuning**.

## **Next Steps**
- **Improve model training**: Add more data augmentation or fine-tune with more epochs.
- **Identify keypoints with high variance**: Consider revising dataset annotations.
- **Compare different models**: Use boxplots to compare multiple architectures.


