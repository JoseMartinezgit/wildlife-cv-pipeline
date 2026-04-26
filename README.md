# Automating Wildlife Identification: A Case Study in Data Leakage and Distribution Shift

By Jose A. Martinez

## Introduction

Camera trap datasets are a powerful tool for wildlife monitoring. Instead of requiring researchers to manually observe animals in the field, remote cameras can capture images whenever motion is detected. These images can then be used to study animal populations, migration patterns, species diversity, and ecosystem health. However, the large volume of images created by these cameras also introduces a major practical problem: manually labeling every image is slow, repetitive, and difficult to scale.

The goal of this project was to build a computer vision pipeline that could automatically classify animal species from wildlife camera trap images using the iWildCam dataset. At first glance, this looks like a straightforward image classification task. Given an image, the model should predict which animal appears in it. In practice, however, wildlife camera data contains several complications that make the problem much more interesting than a normal classification assignment.

The most important lesson from this project was that model accuracy by itself can be misleading. A model can appear to perform extremely well on a validation set while still failing to generalize to the real deployment setting. In this project, changing the way the training and validation sets were split caused validation accuracy to jump from around 57 percent to over 98 percent. That jump was not simply evidence that the model had become a perfect wildlife classifier. Instead, it exposed a deeper issue: the validation setup can accidentally allow data leakage when images from the same camera locations appear in both the training and validation sets.

This project therefore became less about chasing the highest possible accuracy and more about understanding what that accuracy actually means.

## Dataset and Problem Setup

The project used the iWildCam dataset, which contains images captured by remote camera traps. Each image is associated with metadata, including the image file name, the animal category, and information about the camera trap environment. The task was to classify the animal species appearing in each image.

For this implementation, I focused on a smaller balanced subset of the full dataset. Instead of using every species, I selected the five most common animal categories and sampled up to 5,000 images from each class, creating a dataset of approximately 25,000 images total. This made the training process more manageable while still preserving the major engineering challenges of the original problem.

The reduced dataset had two advantages. First, it allowed the model to train within a realistic amount of time using Kaggle’s GPU environment. Second, it created a cleaner experiment where each selected class had a comparable number of images. This made it easier to focus on the impact of data splitting and distribution shift instead of only fighting extreme class imbalance.

Even with this simplified setup, the problem remained challenging because trail camera images are not like standard object classification datasets. The animal is not always centered. Sometimes it is small, partially hidden, blurry, or captured in poor lighting. The background can dominate the image, and the same camera trap may repeatedly capture the same patch of forest, dirt, grass, or shadows.

That last point became the central issue in this project.

## Why Wildlife Camera Data Is Difficult

There were two main challenges in this project: class imbalance and distribution shift.

### Class Imbalance

In wildlife datasets, some animals appear much more often than others. A common animal might trigger a camera thousands of times, while a rare species may only appear in a small number of images. If a model is trained directly on this kind of dataset without adjustment, it can learn a biased strategy. Instead of learning all animals equally, it may over-predict the most common species because that produces a decent accuracy score.

For example, if one species dominates the dataset, a model can achieve a deceptively high accuracy by guessing that species too often. This would not be useful in a real wildlife monitoring system, where identifying rare animals may be especially important.

To reduce this issue, I used class weighting in the loss function and weighted sampling in the training loader. The goal was to make mistakes on less frequent classes more costly and to give each class a more balanced influence during training.

### Distribution Shift

The second challenge was distribution shift. Camera traps are placed in different physical locations. Each location has its own background, lighting, vegetation, camera angle, and environmental conditions. A model trained on one set of camera locations may not perform as well when deployed at a new location.

This matters because the model should ideally learn animal features, such as body shape, color patterns, size, and texture. However, if the dataset is split incorrectly, the model may learn camera-specific background features instead. For example, it might associate a certain species with a particular tree, dirt path, or lighting pattern because that species frequently appeared at one camera.

This creates a major evaluation problem. If images from the same camera location appear in both training and validation, then the validation set is not truly testing whether the model can generalize to a new environment. It is partially testing whether the model recognizes familiar backgrounds.

That is the trap this project investigates.

## Model Architecture

The model used for this project was ResNet18 with pretrained ImageNet weights. ResNet18 is a convolutional neural network architecture that uses residual connections to make training deeper networks easier. Instead of training an entire computer vision model from scratch, I used transfer learning.

Transfer learning is useful because early layers of convolutional neural networks tend to learn general visual features such as edges, corners, textures, and simple shapes. These features are useful across many image tasks, not just the original ImageNet classification task. By starting with pretrained weights, the model already had a strong visual foundation before being adapted to wildlife images.

The original final classification layer of ResNet18 was replaced with a new fully connected layer matching the number of animal classes in my dataset. Since I used five species, the final layer produced five output scores.

Most of the ResNet18 parameters were frozen so that the model did not update every layer during training. I then unfroze the final convolutional block, `layer4`, along with the final fully connected classification layer. This allowed the model to keep the general visual features from ImageNet while still adapting the deeper features to the specific wildlife classification task.

The optimizer used different learning rates for the unfrozen parts of the network. The final convolutional block used a smaller learning rate, while the new fully connected layer used a larger learning rate. This made sense because the final layer was newly initialized and needed more aggressive training, while the pretrained convolutional block only needed fine tuning.

## Data Processing and Augmentation

Each image was resized to 224 by 224 pixels, which matches the expected input size for ResNet18. The images were then converted into tensors and normalized using the standard ImageNet mean and standard deviation values.

I also applied basic data augmentation during training. The transformations included random horizontal flipping and color jitter. Random horizontal flipping helps the model avoid depending too heavily on the direction an animal is facing. Color jitter changes brightness and contrast, which is useful for trail camera data because lighting can vary significantly between daytime, nighttime, shade, and direct sunlight.

The augmentation was not intended to completely solve the distribution shift problem, but it helped make the model less sensitive to small visual changes.

## Handling Class Balance

Even though I sampled the top five classes and limited each class to 5,000 images, I still included class balancing techniques in the pipeline. The training code computed class counts from the training set and used the inverse of those counts as class weights.

These weights were used in two places.

First, they were used in a weighted random sampler. This sampler affected how training batches were created, making it more likely that underrepresented classes would appear during training.

Second, the class weights were passed into the cross entropy loss function. Weighted cross entropy penalizes mistakes differently depending on the true class. If a class is less common, the model receives a larger penalty for misclassifying it. This discourages the model from ignoring less frequent animals.

This was important because accuracy alone can hide poor class-level performance. A model that performs well on common species but poorly on rare species may still have a high overall accuracy. For wildlife classification, that would not be ideal. A better model should be evaluated not only by accuracy, but also by metrics such as macro F1 score and confusion matrices.

## Training Pipeline

The training loop followed a standard PyTorch structure. For each epoch, the model alternated between a training phase and a validation phase. During training, gradients were enabled, the loss was backpropagated, and the optimizer updated the unfrozen parameters. During validation, the model was placed in evaluation mode and gradients were disabled.

For each phase, the code tracked loss and accuracy. After training, the model was evaluated on the validation set, and predictions were collected for additional analysis. I also computed the macro F1 score and generated a confusion matrix. The confusion matrix was especially useful because it showed which classes were being confused with each other rather than only giving a single accuracy value.

I also created a visualization step that displayed validation images alongside their true and predicted labels. This was helpful for understanding the model’s behavior on actual examples instead of only looking at numerical metrics.

## Evaluation Setup: The Most Important Part of the Project

The most important engineering decision in this project was not the choice of ResNet18. It was the way the dataset was split.

In a normal image classification task, it is common to randomly shuffle the dataset and then split it into training and validation sets. This works well when the examples are independent and identically distributed. However, camera trap data violates that assumption because many images are tied to specific physical camera locations.

If a random split is used, images from the same camera trap can appear in both the training and validation sets. That means the model may see the same background, camera angle, and lighting conditions during training and then encounter very similar images during validation. The validation accuracy can become inflated because the model is not being tested on a truly new environment.

A stricter and more realistic evaluation approach is to separate the data by camera location. Under this setup, the training set contains images from one group of cameras, while the validation set contains images from different cameras. This better simulates deployment, where the model may be used on images from camera traps it has never seen before.

Earlier in the project, I used a stricter location-based evaluation. Under that setup, the model achieved around 57 percent validation accuracy. This lower result was not necessarily a failure. In fact, it was more realistic. It showed that generalizing to unseen camera locations is difficult.

Later, I changed the pipeline to use a random shuffled split over the balanced 25,000 image subset. Under this setup, validation accuracy jumped to over 98 percent.

At first, this looked like a major improvement. However, because the evaluation split changed, the two numbers were not measuring the same thing.

The 57 percent result measured performance under a harder and more realistic condition: recognizing animals at unseen camera locations.

The 98 percent result measured performance under an easier condition: recognizing animals when similar camera backgrounds may already appear in the training set.

This difference became the central finding of the project.

## The Random Split Experiment

In the final random-split experiment, the dataset was shuffled and then split into 80 percent training and 20 percent validation. This is a common machine learning workflow, and for many datasets it is reasonable. However, for this dataset it introduced a major risk.

Because the split was random at the image level, images from the same camera trap could appear in both the training and validation sets. This meant the model could learn camera-specific visual patterns. For example, if a certain camera frequently captured one animal species in front of the same tree or dirt path, the model might learn that background as a shortcut.

The resulting validation accuracy was extremely high, reaching over 98 percent.

![Confusion Matrix Showing Random-Split Performance](results/ConfusionMatrix.png)

The confusion matrix showed very strong performance across the selected classes. On the surface, this suggested that the classifier was working extremely well. However, the size of the jump from the location-based result to the random-split result made the result suspicious.

A model usually does not become dramatically better simply because the architecture was fine tuned slightly. A jump from around 57 percent to over 98 percent suggests that something about the evaluation setup changed. In this case, the change was the split strategy.

## Data Leakage and Background Memorization

The random-split result is best understood as a data leakage problem.

Data leakage happens when information from the training set is allowed to influence the validation or test set in a way that would not happen during real deployment. In this project, the leakage did not come from copying labels or accidentally training directly on validation images. Instead, it came from the structure of the data.

Images from the same camera trap often share background features. These features can include trees, bushes, ground texture, shadows, camera angle, lighting, and even the position where animals usually appear in the frame. If the same camera location appears in both training and validation, the model can exploit these repeated patterns.

This does not mean the model completely ignored the animals. It likely learned some animal features too. However, the random split gave the model access to shortcuts that would not be available when deployed at a new camera location. The high validation accuracy therefore did not necessarily prove that the model could generalize to new environments.

A safer interpretation is that the random split allowed the model to use both animal features and camera-specific background cues. The location-based split removed those shortcuts and forced the model to rely more heavily on transferable animal features. That is why the location-based accuracy was much lower.

![Validation Set Visualizer Grid](results/Results1.png)

The visualization grid helped make this issue more concrete. By looking at actual validation images, it became clear that many images contain strong environmental context. In some cases, the background takes up more of the image than the animal. This makes it plausible that a convolutional neural network could use background information as part of its decision process.

## Why the 98 Percent Accuracy Is Not the Full Story

The 98 percent validation accuracy is still an important result, but it should not be interpreted as final proof that the classifier solves wildlife recognition.

Instead, it shows how sensitive machine learning evaluation is to the validation design. If the validation set is too similar to the training set, the model can appear much stronger than it really is. This is especially dangerous in spatial datasets because nearby or repeated locations can create hidden dependencies between examples.

This distinction matters because in real deployment, a wildlife classifier would likely be used on new images from new camera traps. If the model only performs well when it has already seen the same background, then it will not be reliable in the field.

Therefore, the main result of this project is not simply:

> The model achieved over 98 percent accuracy.

A better conclusion is:

> The model achieved over 98 percent accuracy under a random image-level split, but this result was likely inflated by camera-location leakage. A stricter location-based split produced much lower accuracy, showing that true generalization to new camera locations is significantly harder.

That conclusion is more honest and more useful from an engineering perspective.

## Lessons Learned

This project taught several important lessons about applied machine learning.

First, the data pipeline matters as much as the model architecture. It is easy to focus on the neural network, optimizer, learning rate, or number of epochs. Those choices matter, but they cannot fix an evaluation setup that does not match the real problem.

Second, validation accuracy must be interpreted in context. A high validation accuracy is only meaningful if the validation set represents the conditions the model will face after deployment. For wildlife camera data, this means the validation set should contain different camera locations, not just randomly selected images.

Third, distribution shift can be a much harder problem than ordinary classification. The model may perform well when the training and validation images come from similar environments, but performance can drop when the background, lighting, and camera angle change.

Fourth, visual diagnostics are important. Confusion matrices, macro F1 scores, and prediction grids provide more insight than accuracy alone. Looking directly at validation examples can reveal patterns that are hidden in summary metrics.

Finally, this project showed why data leakage is not always obvious. The code did not directly leak labels. The validation images were technically separate from the training images. However, the split still allowed camera-specific background information to appear in both sets. This is a more subtle form of leakage, but it can have a massive effect on performance.

## Future Improvements

There are several ways this project could be extended.

The first improvement would be to make the location-based split the main evaluation method again and keep the random split only as a comparison experiment. This would provide a more realistic estimate of deployment performance.

The second improvement would be to use Grad-CAM or another saliency method to inspect what parts of the image the model focuses on. This would help determine whether the model is looking at the animal or relying heavily on background regions.

The third improvement would be to experiment with stronger augmentation. Random cropping, random erasing, grayscale augmentation, and background perturbation could make it harder for the model to memorize fixed camera environments.

The fourth improvement would be to try object detection or animal cropping before classification. If the model receives an image crop focused on the animal, it may rely less on the background. This could improve generalization across camera locations.

The fifth improvement would be to evaluate with additional metrics such as per-class recall, balanced accuracy, and macro F1 score. These metrics are especially important when dealing with class imbalance because overall accuracy can hide poor performance on individual species.

## Conclusion

This project began as a wildlife image classification task, but it became a case study in data leakage, distribution shift, and evaluation design.

Using a pretrained ResNet18 model, I built a PyTorch pipeline for classifying animal species from iWildCam trail camera images. The pipeline included image preprocessing, data augmentation, class balancing, transfer learning, weighted loss, validation metrics, a confusion matrix, and prediction visualizations.

The most important result was the difference between the location-based evaluation and the random-split evaluation. Under a stricter camera-location split, the model achieved around 57 percent validation accuracy, showing that generalizing to unseen environments is difficult. Under a random image-level split, validation accuracy increased to over 98 percent. However, this increase was not simply proof of a better wildlife classifier. It showed that random splits can allow camera-specific background information to leak into the validation set.

The key takeaway is that high accuracy is only meaningful when the validation setup matches the real deployment scenario. For spatial datasets like iWildCam, random image-level splitting can create misleading results because the model may learn environmental shortcuts. A truly reliable wildlife classifier must be evaluated on unseen camera locations so that it is forced to learn animal features rather than memorize backgrounds.

In the end, this project demonstrated that successful machine learning is not just about building a model that gets a high score. It is about understanding what the score means, identifying when the evaluation is misleading, and designing a pipeline that reflects the real-world problem.
