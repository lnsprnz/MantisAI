# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Keypoint detection training using Tensorflow Object detection API

  - **[Link](https://github.com/prabhakar-sivanesan/Custom-keypoint-detection)**

  - **Objective**: The primary goal of this repository is to provide a comprehensive guide for training keypoint detection models on custom datasets using TensorFlow's Object Detection API. It addresses the gap in resources for detecting keypoints on objects beyond standard human poses or facial landmarks

  - **Methods**: The repository employs the CenterNet architecture with an Hourglass-104 backbone, leveraging transfer learning from models pre-trained on the COCO dataset. The process includes:
  - Data Preparation: 
Collecting images and organizing them into a specified directory structure.
Annotation:
  - Annotation:
Utilizing the COCO Annotator tool to label images with bounding boxes and keypoints, exporting annotations in COCO format.
  - Data Processing: 
Splitting the dataset into training and validation sets, and converting annotations into TFRecord format.
  - Model Configuration: 
Adjusting the TensorFlow model configuration files to align with the custom dataset's specifications.
  - Training: 
Executing the training process using the prepared data and configurations.
  - Inference: 
Applying the trained model to perform keypoint detection on new images or videos.

  - **Outcomes**: By following the guide, users can develop a keypoint detection model tailored to their specific objects of interest. The repository offers scripts and notebooks to facilitate the entire workflow, from data annotation to model inference, enabling effective keypoint detection on custom datasets.
  
  - **Relation to the Project**: Adapting Keypoint Detection to from domains like human pose and face keypoint detection to any other keypoint detection task
 
  - **Implementation Guide**: [Link](https://keras.io/examples/vision/keypoint_detection/)

- **Source 2**: [Title of Source 2]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:

- **Source 3**: [Title of Source 3]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:

- **Other Interesting Stuff**:
    - Keypoint Detection for Identifying Body Joints using TensorFlow [Link](https://dl.acm.org/doi/pdf/10.1145/3590837.3590948?casa_token=ElPrKE9p2k4AAAAA:hX18DHmKRVazLsZ6gusm59i3RhDYuEBwpQMpLkTn8ao77wVEk6DN8oLxmVHLP09YJsLdCEZexjiL)
    - 
