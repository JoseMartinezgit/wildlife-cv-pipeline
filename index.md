# Automating Wildlife Identification with Deep Learning

By Jose A. Martinez, Andrew Choi, Pavan Patel, Seth Sandberg

## Introduction

Wildlife camera traps are an important tool for studying animal populations and monitoring natural environments. These cameras are placed in outdoor locations and automatically capture images when motion is detected. This makes them useful for collecting large amounts of wildlife data without requiring constant human observation.

However, the large number of images created by trail cameras also creates a major data science challenge. Manually sorting through thousands of images and labeling the animal species in each one is time-consuming. A machine learning model that can automatically identify animals from camera trap images would make this process much faster and more scalable.

For this project, our team built a computer vision pipeline using the iWildCam dataset. The goal was to classify animal species from trail camera images using a deep learning model. Our final system used transfer learning with a pretrained ResNet18 model, balanced data sampling, weighted loss, image augmentation, and validation visualizations to evaluate performance.

Our model achieved strong validation performance, reaching over 98 percent accuracy on the selected balanced dataset. This showed that our pipeline was able to learn meaningful visual patterns associated with the target animal classes and classify validation images accurately.

## Project Goal

The main goal of this project was to design and evaluate a deep learning pipeline for automated wildlife identification. The model takes an image from a remote trail camera and predicts which animal species appears in the image.

This problem is a strong example of applied data science because it involves more than simply training a neural network. The dataset contains real-world image variation, including differences in lighting, background, camera angle, animal pose, image quality, and class frequency. Because of this, the project required careful preprocessing, model design, training strategy, and evaluation.

Our project focused on the following objectives:

1. Load and process metadata from the iWildCam dataset.
2. Build a balanced image classification dataset from the most common species.
3. Train a convolutional neural network using transfer learning.
4. Address class balance using weighted sampling and weighted loss.
5. Evaluate the model using accuracy, macro F1 score, a confusion matrix, and prediction visualizations.
6. Interpret the model’s performance and understand the effectiveness of the pipeline.

## Dataset

The dataset used in this project was the iWildCam camera trap dataset. This dataset contains images captured by remote wildlife cameras. Each image is associated with metadata, including the image file name and the animal category label.

Because the full dataset is large and contains many animal categories, our team focused on a smaller subset for the final experiment. We selected the top five most common species and sampled up to 5,000 images from each class. This produced a balanced dataset of approximately 25,000 images.

Using this balanced subset made the training process more manageable while still preserving the main challenge of wildlife image classification. The model still had to distinguish between different animal species under natural camera trap conditions, but the dataset was small enough to train efficiently in the Kaggle GPU environment.

Balancing the dataset also helped prevent the model from being dominated by one extremely common species. In many real-world datasets, some classes appear much more frequently than others. If this imbalance is not handled, the model may learn to over-predict the most common classes. By using a more balanced subset, our model had a better opportunity to learn the visual features of each animal category.

## Data Preprocessing

The first step in our pipeline was loading the iWildCam metadata. The image metadata, annotation metadata, and category metadata were loaded from the dataset’s JSON annotation file. These were converted into pandas DataFrames so that the image paths, labels, and category names could be processed more easily.

After loading the metadata, we filtered the dataset to keep only the top five most common animal categories. We then sampled up to 5,000 images from each class and created a label mapping that converted each category ID into a numerical class index.

Each image was loaded using a custom PyTorch `Dataset` class. This class retrieved the image file path, opened the image using PIL, converted it to RGB format, applied the selected transformations, and returned the image tensor along with its label.

The image transformations included:

- resizing each image to 224 by 224 pixels,
- random horizontal flipping,
- color jitter for brightness and contrast,
- conversion to a PyTorch tensor,
- normalization using ImageNet mean and standard deviation values.

These transformations were important because ResNet18 expects images in a standard size and normalized format. The augmentation steps also helped the model become more flexible by exposing it to small variations in image appearance during training.

## Model Architecture

The model used for this project was ResNet18 with pretrained ImageNet weights. ResNet18 is a convolutional neural network architecture that uses residual connections to train deeper networks more effectively. Instead of training a neural network from scratch, we used transfer learning.

Transfer learning was useful because the lower layers of a pretrained image model already contain general visual knowledge. These layers can detect basic features such as edges, textures, shapes, and patterns. Since wildlife classification is also an image recognition task, starting from pretrained ImageNet weights gave our model a stronger foundation than random initialization.

To adapt ResNet18 to the wildlife classification task, we replaced the original final fully connected layer with a new linear layer that produced outputs for the five selected animal classes.

Most of the pretrained ResNet18 layers were frozen. This means their weights were not updated during training. Freezing the earlier layers helped preserve the general visual features learned from ImageNet and reduced the amount of training required.

The final convolutional block, `layer4`, was unfrozen so that the model could learn more task-specific features from the wildlife images. The final classification layer was also trained from scratch. This gave the model a balance between using pretrained general image features and adapting to the specific animal species in the dataset.

## Training Strategy

The model was trained using the Adam optimizer. We used two different learning rates:

- a smaller learning rate for the unfrozen ResNet18 `layer4`,
- a larger learning rate for the new fully connected classification layer.

This strategy made sense because the final classification layer was newly initialized and needed to learn quickly, while the pretrained convolutional layer only needed fine tuning.

The model was trained for five epochs. During each epoch, the pipeline ran both a training phase and a validation phase. In the training phase, the model updated its weights using backpropagation. In the validation phase, the model was evaluated without updating weights.

For each phase, the code tracked loss and accuracy. This made it possible to monitor whether the model was learning over time and how well it performed on validation images.

## Handling Class Balance

Even though the final dataset was balanced by sampling from the top five classes, we still included additional class balancing methods in the training pipeline. This made the pipeline more robust and better suited for datasets where perfect balance is uncommon.

First, we calculated the number of training examples in each class. Then we computed class weights using the inverse of the class counts. Classes with fewer examples received larger weights, while classes with more examples received smaller weights.

These weights were used in two ways.

The first method was a `WeightedRandomSampler`. This sampler controlled how training batches were created. Instead of sampling all images uniformly, it used sample weights to help balance class representation during training.

The second method was weighted cross entropy loss. Cross entropy is a standard loss function for classification, but the weighted version allows some classes to have a larger effect on the loss. This helps prevent the model from ignoring less frequent classes.

Together, weighted sampling and weighted loss helped ensure that the model learned from all selected animal categories instead of becoming biased toward one class.

## Evaluation Method

After training, the model was evaluated on the validation set. The evaluation step collected all predicted labels and true labels, then calculated performance metrics.

The main metric was validation accuracy, which measured the percentage of validation images classified correctly. Our model achieved over 98 percent validation accuracy on the balanced selected dataset, showing that the trained ResNet18 pipeline was highly effective at distinguishing between the five animal categories.

We also calculated the macro F1 score. Macro F1 is useful because it treats each class equally instead of allowing larger classes to dominate the score. This is especially important in animal classification tasks, where each species should be identified reliably.

In addition to numerical metrics, we generated a confusion matrix. The confusion matrix shows how often each true class was predicted as each possible class. This made it easier to see whether the model was confusing certain animals with each other.

![Confusion Matrix Showing Model Performance](results/ConfusionMatrix.png)

The confusion matrix showed that the model performed strongly across the selected classes. Most predictions appeared along the diagonal, meaning the predicted labels matched the true labels for the majority of validation examples.

## Visualizing Model Predictions

Numerical metrics are useful, but they do not always show what the model is doing on actual images. To better understand the model’s predictions, we created a visualization grid of validation images.

The visualization step selected a batch of validation images, passed them through the trained model, and displayed eight examples in a grid. Each image was shown with its true label and predicted label. Correct predictions were shown in green, while incorrect predictions were shown in red.

![Validation Set Visualizer Grid](results/Results1.png)

This visualization made the evaluation more interpretable. Instead of only seeing an accuracy number, we could inspect specific examples and confirm that the model was correctly identifying animals in real images.

The visualizer was also useful for debugging. If the model made mistakes, this grid could help reveal whether the error was understandable. For example, some animals may appear very small, partially hidden, or difficult to distinguish from the background. Looking at examples directly gives more context than metrics alone.

## Results

The final model achieved strong performance on the validation set, with validation accuracy rising above 98 percent. This result showed that the ResNet18 transfer learning approach was effective for the selected wildlife classification task.

Several design choices contributed to this performance:

1. **Transfer learning:** Starting with ImageNet pretrained weights allowed the model to use general visual features from the beginning of training.

2. **Fine tuning:** Unfreezing the final convolutional block allowed the network to adapt higher-level visual features to the animal classification problem.

3. **Balanced dataset construction:** Sampling the top five classes evenly helped reduce class imbalance and made the classification task more stable.

4. **Weighted sampling and weighted loss:** These methods further helped the model treat the selected classes more evenly during training.

5. **Data augmentation:** Random flips and color jitter exposed the model to variation in image appearance.

6. **Visual diagnostics:** The confusion matrix and prediction grid made it possible to evaluate the model beyond a single accuracy number.

Overall, the results showed that our pipeline was able to learn useful visual representations for wildlife classification and apply them successfully to validation images.

## Discussion

This project demonstrated how a pretrained convolutional neural network can be adapted to a real-world image classification problem. The iWildCam dataset is more complex than a simple classroom dataset because the images come from outdoor camera traps and contain natural variation in lighting, background, animal position, and image quality.

Our model handled these challenges well on the selected balanced dataset. By using ResNet18, the pipeline avoided the need to train a large model from scratch. By fine tuning the final convolutional block, the model was able to adjust to wildlife-specific features. By using weighted loss and weighted sampling, the training process remained focused on all selected animal categories.

One of the most important parts of the project was building the full pipeline from metadata loading to final evaluation. The project was not just about calling a pretrained model. It required processing the annotation files, constructing a usable dataset, defining a custom PyTorch dataset class, applying transformations, building dataloaders, modifying the model architecture, training the network, computing evaluation metrics, and visualizing predictions.

This made the project a complete applied machine learning workflow.

## Software Engineering Perspective

From a software engineering perspective, the project emphasized the importance of building a clear and reproducible pipeline. Each part of the code had a specific role.

The metadata loading section handled the raw iWildCam annotations and converted them into DataFrames. The filtering section selected the animal classes used in the experiment. The custom dataset class connected the metadata to the actual image files. The transformations standardized and augmented the input images. The dataloaders handled batching and sampling. The model section defined the ResNet18 transfer learning setup. The training loop handled optimization and evaluation. Finally, the visualization section produced interpretable outputs for the final report.

This structure made the code easier to understand and modify. For example, the number of classes, the image transformations, the model architecture, and the training parameters could all be adjusted without rewriting the entire pipeline.

The project also showed how machine learning systems require both modeling decisions and engineering decisions. The accuracy of the final model depended not only on ResNet18, but also on how the dataset was prepared, how the classes were balanced, how the loss function was defined, and how the results were evaluated.

## Future Improvements

Although the final model performed strongly, there are several ways the project could be extended.

One future improvement would be to train on more animal categories from the full iWildCam dataset. This project focused on the top five classes to keep training manageable, but expanding to more species would make the classifier more comprehensive.

Another improvement would be to train for more epochs and experiment with additional learning rates. The model already performed well after five epochs, but additional tuning could help improve performance further or make training more stable.

A third improvement would be to try larger architectures, such as ResNet50 or EfficientNet. These models may be able to learn more detailed visual features, although they would require more computation.

Another possible extension would be to add more data augmentation. Random cropping, rotation, random erasing, or grayscale augmentation could make the model more robust to different trail camera conditions.

Finally, the pipeline could be extended into a more complete wildlife monitoring tool. For example, the model could be connected to a dashboard that displays predicted species counts, confidence scores, and example images for researchers to review.

## Conclusion

This project developed a complete deep learning pipeline for automatic wildlife identification using the iWildCam dataset. The final system used a pretrained ResNet18 model, transfer learning, class balancing, weighted loss, data augmentation, and visual evaluation tools.

Our model achieved over 98 percent validation accuracy on the selected balanced dataset of five animal classes. This strong performance showed that the pipeline was able to learn meaningful visual patterns from trail camera images and classify validation examples accurately.

Beyond the final accuracy, the project demonstrated the full process of applied machine learning: preparing the data, designing the model, handling class balance, training the network, evaluating performance, and interpreting results through visualizations.

Overall, this project shows how computer vision can be used to automate wildlife identification and reduce the amount of manual effort required to analyze trail camera data. With further expansion to more species and additional evaluation settings, this type of pipeline could become a useful tool for large-scale wildlife monitoring.
