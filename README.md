## MantisAI - Identify pivot points between insect leg segments

## Repository Link

[https://github.com/lnsprnz/MantisAI)]

## Description

The project will involve analyzing highspeed videos of the movement of the legs of an insect. It will be done on a black and white image stack (a couple thousand per video), the insect will be a praying mantis and the videos show the mantis catching a prey item (fly). The goal is, that the model can find the pivot points between the leg segments, so that from there it would be possible to calculate different parameters of the strike (e.g. angular velocities, angles, tangential velocities etc.)

Currently the pivot points are set manually every 10 frames and are interpolateted. This is a huge manual effort that is very time consuming.

As an introduction and overview of the parts of the animal we will be speaking of, the general habitus can be seen in Figure 1.


![Figure-1](https://github.com/user-attachments/assets/4fc0f98a-ead5-4dde-9b66-af0a0ee3ee9d)

Figure 1 - Adult, male Hierodula majuscula, lateral view (a) and latero-dorsal view (b). a, abdominal; d, dorsal; f, frontal; v, ventral.

There are 5 points of interest on this animal that we would like to track: 1) the body, at the base of the forewings (blue), (2) the coxal base (green), (3) the trochanter-coxa joint (orange), (4) the femur-tibia joint (purple) and (5) the base of the tarsus (Figure 2). From there, in a post processing step, the angles and of the joints and the corresponding parameters of the motion in all frames can be determined: body-coxa (BC) angle, coxa-trochanter-femur (CT) angle, and femur-tibia (FT) angle (Figure 2).

![RawAngle_Velocity_TrackingReihe](https://github.com/user-attachments/assets/e550c6db-f7a7-4b1c-adfc-5985d1709113)

Figure 2 - 

**Update finding model**
Our videos conatain approximately 1000-2000 images, depending on the file. They are in format 1024x1024 - it is possible to crop a large empty party of all images to reduce the size. In the end, we want to find the pivot points of specific joints in the animal and get their coordinates for every frame. These coordinates should be exported as a text file for further analysis.
Ideally, the process should be based on unsupervised object detection, but we also found other possible solutions that may also include some manual key point labeling.
 
In our literature review, we did not find one perfect solutions, but many inspirations to try out and maybe combine. We now try to identify the best possible model that we could use as a starting point to build on.

Here, we decided to have a closer look at three different approaches.

1) Mediapipe from Google. The tool is apparently best trrained for human parts, but the general approach would be ideal for our project. Here, the advantage would be, that the model is already designed to find pivot points on its own.
- Prerequisite: Need to annotate own images on pivot points. Need to adjust convolutional network.

2) Unsupervised detection using a preexisting repository of Antonilo, including heatmap generation of moving objects. The resulting heatmap data could be useful to calculate pivot points.
- Unclear documentation and running of script.

3) Keypoint_Detection
- Prerequisite: Need to annotate own images on pivot points.
 

### Task Type

2D Keypoint Detection

### Results Summary

- **Best Model:** [MobileNetV2 backbone with additional CNN layers]
- **Evaluation Metric:** [OKS and Euclidian Distance]
- **Result:** **[Boxplot](3_Model/MantisAI_Results_Graphics/BoxplotKeypointsDistance.png)**

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/README.md)**
2. **[Model Definition and Evaluation](2_Model/README.md)**
4. **[Presentation](3_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/D2_AN04_S1_C001H001S0001000228_Beinsegmente+Pivotpoints (2).tiff)
