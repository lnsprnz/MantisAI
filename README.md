## MantisAI - Identify pivot points between insect leg segments

## Repository Link

[https://github.com/lnsprnz/MantisAI)]

## Description

The project will involve analyzing highspeed videos of the movement of the legs of an insect. It will be done on a black and white image stack (a couple thousand per video), the insect will be a praying mantis and the videos show the mantis catching a prey item (fly). The goal is, that the model can find the pivot points between the leg segments, so that from there it would be possible to calculate different parameters of the strike (e.g. angular velocities, angles, tangential velocities etc.)

Currently the pivot points are set manually every 10 frames and are interpolateted. This is a huge effort

### Task Type

2D Keypoint Detection

### Results Summary

- **Best Model:** [Name of the best-performing model]
- **Evaluation Metric:** [e.g., Accuracy, F1-Score, MSE]
- **Result:** [e.g., 95% accuracy, F1-score of 0.8]

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
