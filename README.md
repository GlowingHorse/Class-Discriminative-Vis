# Grouped class discriminative feature visualization
For the manuscript "Group Visualization of Class-Discriminative Features"

## Introduction
A tensorflow based implementation of the method of Group Visualization of Class-Discriminative Features.
You would better to install necessary libraries listed in the "requirement.txt" file at first.

## How to use
1. Install all necessary library and clone/download this repository.

2. Run "GroupClassDiscVis.py" and the results of the 1st 3rd experiments in the paper could be generated.

3. With the attribution computing method, you can test other layers, classes, and images.

4. Some description for code in main dir. 
   * *data* is for placing testing images, for GoogleNet, input size is 224.
   * *dog_cat224* is the dir named by the image name for saving generated visual results and computed attribution results (those .txt files). 
   * *utils* is for main codes to compute attribution, group features, loss function, and some activation maps handling operators.

5. The codes for generating adversarial samples and experiment results about these samples will be uploaded later.

6. When you try different networks that are provided in the library "lucid", the transform methods and some random image preconditioning settings should be changed for a good visualization. We suggest to read more information about activation maximization methods from the codes of the "lucid" library. They provided many useful loss functions, regularizations, and preconditioning methods.


## Try it
1. It will be fun to test different images for analyzing the network.
