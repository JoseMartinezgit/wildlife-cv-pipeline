# Automating Wildlife Identification: A Case Study in Data Leakage and Distribution Shift

By Jose A. Martinez

## Introduction
The iWildCam dataset presents a unique computer vision challenge. The goal is to automate the identification of various animal species captured on remote camera traps. Building a robust machine learning pipeline for this task requires navigating significant data irregularities and ensuring the model generalizes to new environments.

## The Engineering Challenges
Two primary hurdles exist in wildlife image classification. First is the extreme class imbalance. Certain species trigger the cameras thousands of times while rare animals might only appear a handful of times. Second is the distribution shift. Camera traps are deployed across diverse environments meaning the lighting, foliage, and angles change drastically from one location to another. Designing a system that learns the animal rather than the background is paramount.

## System Architecture and Pipeline
To tackle this, the pipeline utilizes a ResNet18 architecture with pretrained ImageNet weights. The transfer learning process was fine tuned by unfreezing the final convolutional block, allowing the model to learn specific animal textures. 

To combat the class imbalance, a weighted cross entropy loss function was implemented. This forced the neural network to heavily penalize misclassifications of rare species, ensuring all animals were prioritized equally during training regardless of their frequency in the dataset.

## The Evaluation and The Trap
The initial model evaluation maintained strict isolation between camera locations. The training set and validation set contained images from completely different cameras. Under these conditions, the validation accuracy hovered around 57 percent. This highlighted the severe distribution shift as the model struggled to identify familiar animals in unfamiliar environments.

To demonstrate the impact of spatial data isolation, an experiment was conducted where a balanced dataset of 25,000 images was randomly shuffled before the validation split. The performance immediately skyrocketed to over 98 percent accuracy.

![Confusion Matrix Showing Data Leakage](results/ConfusionMatrix.png)

## Understanding Data Leakage
This massive jump in accuracy is a textbook example of data leakage. By randomly shuffling the dataset, images from the exact same camera trap ended up in both the training and validation sets. The ResNet18 model did not suddenly become better at identifying the animals. Instead, it memorized the static backgrounds of the camera traps. When tested on the validation set, it simply recognized the trees and shadows it had already seen during training.

![Validation Set Visualizer Grid](results/Results1.png)

## Conclusion
This project illustrates a fundamental principle in Software Engineering and Systems design: a machine learning model is only as robust as its data pipeline. Achieving high accuracy on a validation set is meaningless if the data splits do not reflect deployment conditions. For remote sensing and spatial datasets, strict separation of physical locations is non negotiable to ensure true generalization.
