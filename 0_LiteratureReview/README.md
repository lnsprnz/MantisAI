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

- **Source 2**: [StanfordExtra]

  - **[GitRepo](https://github.com/benjiebob/StanfordExtra)**
  - **[Paper](https://arxiv.org/abs/2007.11110)**
  - **[Implementation Guide](https://keras.io/examples/vision/keypoint_detection/)**
  - **Objective**:  
  The primary goal of this repository is to provide a dataset of 12,000 images of dogs annotated with 2D keypoints and segmentation masks, enabling research in 3D animal reconstruction and other computer vision tasks.

  - **Methods**:  
  The repository processes and enriches the Stanford Dogs Dataset to include:  
    - Data Collection: Curating images from the Stanford Dogs Dataset.  
    - Annotation: Labeling images with 2D keypoints and segmentation masks, focusing on the primary dog in images with multiple dogs.  
    - Data Format: Consolidating all annotations, segmentations, and metadata into a single JSON file for ease of access.  
    - Usage Demonstration: Providing a Jupyter Notebook (`demo.ipynb`) to showcase the dataset’s utilization and visualization.

  - **Outcomes**:  
  The repository offers a well-annotated dataset specifically for computer vision tasks involving animals, such as 3D reconstruction and pose estimation. Its structured format and demonstration notebook facilitate easy adoption in research workflows.
  - **Relation to the Project**: Keypoint Detection on Animal Dataset. well described.

- **Source 3**: [Title of Source 3]

  - **[Paper](https://dl.acm.org/doi/pdf/10.1145/3590837.3590948?casa_token=ElPrKE9p2k4AAAAA:hX18DHmKRVazLsZ6gusm59i3RhDYuEBwpQMpLkTn8ao77wVEk6DN8oLxmVHLP09YJsLdCEZexjiL)**

  - **Objective**:  
The primary goal of this paper is to explore and compare state-of-the-art architectures for human pose estimation through keypoint detection using TensorFlow. The research aims to evaluate the effectiveness of different models like DeepPose, HRNet, and OpenPose for detecting body joints in humans, focusing on their performance across multiple datasets.

  - **Methods**:  
    - **Models**:  
      - **DeepPose**: A regression-based approach using convolutional neural networks (CNN) to estimate key body joints.
       - **HRNet**: Maintains high-resolution representations throughout the network for superior pose estimation accuracy.
       - **OpenPose**: A bottom-up architecture that detects all body joints in an image and groups them into individuals.
  
    - **Dataset**:  
      - Utilizes the COCO and MPII datasets, which provide annotated keypoints for human pose detection.  
     - Preprocessing includes cleaning and structuring annotations into accessible formats using libraries like mmPose and CocoDataset.

    - **Evaluation Metrics**:  
      - **PDJ (Percentage of Detected Joints)**: Measures the percentage of keypoints detected within a threshold distance.
      - **OKS (Object Keypoint Similarity)**: Combines distance, scale, and keypoint-specific thresholds for evaluation.
      - **mAP (Mean Average Precision)**: Averages precision across all keypoints and datasets.

    - **Implementation**:  
      - Conducted experiments using mmPose for DeepPose and HRNet implementations and TensorFlow for OpenPose.  
      - Evaluations were performed on the entire datasets, and visual inference was used to demonstrate real-world performance.

  - **Outcomes**:  
    - HRNet consistently demonstrated the highest accuracy across all metrics, especially on large body joints like shoulders, with slight struggles on smaller joints like ankles.  
    - DeepPose and OpenPose provided faster inference times but with reduced precision compared to HRNet.  
    - The architectures showed varying performance, with HRNet achieving the best balance of speed and accuracy, followed by DeepPose and OpenPose.  
    - The results emphasize HRNet's superiority in maintaining spatial resolution and the need for optimization in detecting smaller body joints.

  - **Relation to the Project**: Overview of different Keypoint detection models. Downside: no repo attached...

- **Other Interesting Stuff**:
    - Keypoint Detection for Identifying Body Joints using TensorFlow [Link](https://dl.acm.org/doi/pdf/10.1145/3590837.3590948?casa_token=ElPrKE9p2k4AAAAA:hX18DHmKRVazLsZ6gusm59i3RhDYuEBwpQMpLkTn8ao77wVEk6DN8oLxmVHLP09YJsLdCEZexjiL)
    - A lightweight keypoint matching framework for insect wing morphometric landmark detection [Link](https://www.sciencedirect.com/science/article/abs/pii/S1574954122001443)
    - Animal Pose Estimation: Fine-tuning YOLOv8 Pose Models [Link](https://learnopencv.com/animal-pose-estimation/) [Video](https://www.youtube.com/watch?v=kb03ufEkOdA)
      Simalar approach as the StanfordExtra Paper/Repo with the same Dataset using yolo model & annotation instead of coco annotation which is possible easier to create from existing files of Fabian.
