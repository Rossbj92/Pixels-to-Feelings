# Pixels to Feelings: A Comparison of Visual Sentiment Methods

In this project, I sought to build a generalizable model for visual sentiment tasks. I compared 3 methods:
1. Curriculum learning using a convolutional neural network (CNN) and the WEBEmo dataset
2. Fine-tuning a pre-trained CNN on the Deep Emotion dataset
3. Using an ImageNet pre-trained model for feature extraction


## Replication Instructions

1. Clone repo
2. Install package requirements in ```requirements.txt```.
3. Gather WEBEmo images with the [image-gather](notebooks/image-gather.ipynb) notebook and train the model with the [webemo](notebooks/webemo.ipynb) notebook.
4. Gather Deep Emotion images and train the model with the [deep-emotion](notebooks/deep-emotion.ipynb) notebook.
5. Gather UnBiasedEmo images and train all models with the [experiment](notebooks/deep-emotion.ipynb) notebook.

The datasets (i.e., WEBEmo, Deep Emotion, and UnBiasedEmo) are too large to upload, so the images must be downloaded locally for the training code to run.

## Directory Descriptions

```Data```
- ```csvs``` - csv files corresponding to WEBEmo URLs and labels.
- ```logs``` - training logs for the WEBEmo, Deep Emotion, and ImageNet models in the ```experiment``` notebook.

```Notebooks```
- [deep-emotion](notebooks/deep-emotion.ipynb) - obtains Deep Emotion images and trains Deep Emotion model
- [experiment](notebooks/deep-emotion.ipynb) - obtains UnBiasedEmo images and trains the 3 final models
- [image-gather](notebooks/image-gather.ipynb) - obtains WEBEmo images
- [webemo](notebooks/webemo.ipynb) - trains a curriculum learning model on the WEBEmo dataset
- [visualizations](notebooks/visualizations.ipynb) - produces validation losses/accuracies plots and confusion matrices

```Models```
- Holds trained models - currently empty due to model sizes and will populate after training locally

```Reports```
- ```presentation``` - pdf and ppt format of final project presentation
- ```figures``` - images from visualizations notebook

```src```
- Contains code for functions used in the notebooks

## Conclusions

Test accuracies for the 3 methods on the UnBiasedEmo dataset were:
1. WEBEmo – 72.74%
2. Deep Emotion - 66.50%
3. ImageNet – 59.61%

The WEBEmo trained model had the most evidence of generalizability. For future visual sentiment tasks, using a WEBEmo curriculum-based model for transfer learning deserves consideration for fine-tuning.

## Methods

- Deep learning
- Convolutional neural networks
- Pytorch
- Curriculum learning
- Transfer learning
- Cloud GPU training


