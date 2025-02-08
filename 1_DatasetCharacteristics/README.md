# Dataset Characteristics

## Dataset Overview

In our project, we used black and white images from high-speed video recordings of a praying mantis, hunting (main README Figure 1). The animals were placed orthogonally to the camera, with a prey item (fly) on the left hand side. First, the mantis will approach the prey item and then initiate the predatory strike. During the strike, the animal is rapidly extending its forelegs to get its tibia behind the prey, to lock it between femur and tibia (parts of the insects leg can be seen in Figure 2). As a first trial, we are analizing only images of one species (_Hypsicorypha gracilis_, Figure 1).

All high-speed video recordings were taking with 4000 frames per second and all images are initially 1024x1024 in size and in .tif format. Usually, one video of a predatory strike has approximately 1000-2000 images that are relevant to characterize the motion of the predatory strike. In our model, we used 220 images from different videos, showing the mantis in different postures during the striking sequence.

To generate annotated images for the training and validation process, we used Roboflow (https://roboflow.com/). Using the free parts of the software, we annotated the described points that need to be tracked (main README) in all images. To incorporate the data in our model, we exported the whole dataset as a .jason file, that contains all relevant information of the spatial position of the points of interest.

## Missing Values

Missing values were no problem in our current approach, but could be influential in umcoming different approaches. In some videos, individual tracking points may not be visible the entire time. Usually, those videos are discarded, and are therefore not a cause for missing values.
As the video camera is considerably old and well used, it happens from time to time, that single images, or 2-3 concecutive images are only black and white pixels.
For now, we ignored this type of images. In the future, it would be imaginable to include a consecutive series of images for every frame that is being trained. In this case, a different approach to handle missing data would be necessary.

## Feature Distribution

At the moment, we only used the annotated points on the images as features (5). In the future, the described approach, using a consectutive series of images for training would be imaginable, to increase the available features.

## Possible Biases

At the moment, the selection of the images for the annotation process happens manually, and is probably a source for biases. We tried to chose images from different striking postures, but overall, some postures may be overrepresented. In general, the strike itself happens very fast ( approximately 60 ms). In the obtained high-speed videos, the praying mantis is therefore most of the time in a relaxed posture. Therefore, even if the complete image stack of the videos would have been used, the relaxed posture would be overrepresented. Fur further work on that project, this problem would have to be addressed.
