# Grouped class discriminative feature visualization
For the manuscript "Group Visualization of Class-Discriminative Features"

## Introduction
A `tensorflow` based implementation of the method of Group Visualization of Class-Discriminative Features.
You would better to install necessary libraries listed in the "requirement.txt" file at first.

## How to use
1. Install all necessary library and clone/download this repository.

2. Run *GroupClassDiscVis.py* and the results of the 1st and 3rd experiments in the manuscript could be generated.

3. With the attribution computing method, you can test other layers, classes, and images. The grouping methods could be changed with the clustering methods provided by the library [`sklearn`](https://scikit-learn.org/stable/modules/clustering.html). We also provide two simple *class*es in the file *./utils/utils.py*, it should be easy to extended for trying other grouping features methods like clustering and matrix decomposing.

4. Some description for code in main dir. 
   * *data* is for placing testing images, for GoogleNet, input size is 224. The two `matlab` files are used for mosaic figures in the paper.
   * *dog_cat224* is the dir named by the image name for saving generated visual results and computed attribution results (those *.txt* files). 
   * *utils* is for main codes to compute attribution, group features, loss function, and some activation maps handling operators.

5. Run *GroupKernelVisShapResNet.py* to generate visualization results in ResNet-50.

6. The codes for generating adversarial samples will be uploaded later. We also plan to rearrange the structure of the codes for easier reading further.

7. When you try different networks that are provided in [`TensorFlow-Slim`](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models), the transform methods and some random image preconditioning settings should be changed accordingly for good visualization. We suggest to read more about activation maximization methods from the codes of [`lucid`](https://github.com/tensorflow/lucid). They have integrated many useful loss functions, regularizations, and preconditioning methods from a lot of literatures about activation maximization.

## Try it
1. It will be fun to visualize what features are extracted from different objects in the image.
2. If you have any question, please contact with us by email or comment in [**Issues**](https://github.com/GlowingHorse/class-discriminative-vis/issues).
