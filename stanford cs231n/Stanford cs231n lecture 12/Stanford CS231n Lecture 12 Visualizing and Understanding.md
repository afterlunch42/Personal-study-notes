## Stanford CS231n Lecture 12: Visualizing and Understanding

### 1. What's going on inside ConvNets?

**First Layer**: Visualize Filters

![屏幕截图(3)](屏幕截图(3).png)

looking like something with oriented edges and opposing colors.

**filters/kernels**:

we can visualize filters at higher layers, but not that interesting.

visualize method:

the input is this has 16 dimensions in depth, and each convolutional filter is 7*7，and is extending along the full depth so has 16 elements. Then we have 20 such of these convolutional filters, that are producing the next layer.

For this single $16\times 7\times 7$ convolutional filter, we can spread out those $16\times7\times7$ planes of the filter into a $16\times7\times7$ gray scale images, which is these little gray scale images, can show us what are the weights in one of the convolutional filters of the second layer.

![屏幕截图(4)](屏幕截图(4).png)

All the visualization on this on the precious slide has the scale the weights to the 0 to 255 range.

**Last layer**：

Run the network on many images, collect the feature vectors.

We can use the *Nearest Neighbors* to  visualize the last layer:

![屏幕截图(5)](屏幕截图(5).png)

Last layer is capturing some of those semantic content of these images.

We can also use the *dimensionality reduction* to visualize the last layer:

![屏幕截图(6)](屏幕截图(6).png)

such as Principle Component Analysis(PCA) and t-distributed stochastic neighbor embeddings(t-SNE).

![屏幕截图(7)](屏幕截图(7).png)

visualizing activations:

![屏幕截图(8)](屏幕截图(8).png)

**Occlusion Experiments**:

![屏幕截图(9)](屏幕截图(9).png)

Saliency Maps:

Computing the gradient of the predicted class score with respect to the pixels of the input image.

In this sort of first order approximation sense, for each input and each pixel in the input image, if we wiggle(扰动) that pixel a little bit, then how much will the classification score for the class change.

This is a matter of Which pixels in the input matter for the classification.

![屏幕截图(10)](屏幕截图(10).png)

**Saliency maps**: segmentation without supervision

![屏幕截图(11)](屏幕截图(11).png)

**Intermediate Features via backprop**:

compute gradient of neuron value with respect to image pixels.

![屏幕截图(12)](屏幕截图(12).png)

![屏幕截图(13)](屏幕截图(13).png)

Visualizing CNN features:**Gradient Ascent**

![屏幕截图(14)](屏幕截图(14).png)

Fix the weight of out trained convolutional network and instead synthesizing image by performing Gradient Ascent on the pixels of the image to try and maximize the score of some intermediate neuron or of some class.

![屏幕截图(15)](屏幕截图(15).png)

Better regularizer: Penalize L2 norm of image; also during optimization periodically

(1) Gaussian blur image

(2) Clip pixels with small values to 0

(3) Clip pixels with small gradients to 0



Fooling Images/ Adversarial Examples:

(1) start from an arbitrary image

(2) pick an arbitrary class

(3) modify the image to maximize the class

(4) repeat until network is fooled

![屏幕截图(16)](屏幕截图(16).png)

**DeepDream**: 

Amplify existing features

![屏幕截图(17)](屏幕截图(17).png)

code:

![屏幕截图(18)](屏幕截图(18).png)

**Feature Inversion**:
![屏幕截图(19)](屏幕截图(19).png)