### The Fluorescent Veil: A Stealthy and Effective Physical Adversarial Patch Against Traffic Sign Recognition

This is the official implementation in the paper **The Fluorescent Veil: A Stealthy and Effective Physical Adversarial Patch Against Traffic Sign Recognition** in Python.



### Quickstart

1. Install the packages used in our scheme

   `pip install -r ./requirement.txt`

2. Set the following parameters in `fipatch_attack.py`:

   ```python
   dataset:          datasets for model training and adversary attacks.
                     'ctsrd'; 'gtsrb'
   model_name:       black-box model.
                     'resnet50'; 'resnet101'; 'vgg13'; 'vgg16'.         —— in ctsrd dataset.
                     'cnn'; 'inceptionv3'; 'mobilenetv2'; 'googlenet'.  —— in gtsrb dataset.
   perturb_radius:   radius of the added perturbation.
   circle_number:    number of fluorescent circles.
   n_restarts:       n-random-restarts strategy for the particle swarm optimization.
   location:         whether to record the perturbation center of a successful attack.
   color_test:       whether to fix the perturbation color.
   color_used:       if color_test is set to True, set the perturbation color
   save_folder:      record images of failed attacks
   ```

3. Run the `fipatch_attack.py`

   `python fipatch_attack.py`

### Demonstration videos

- [FIPatch](https://sites.google.com/view/fipatch-attack/home)


### What is the FIPatch?

- In this work, we design a stealthy and effective physical adversarial patch using fluorescent ink (FIPatch). Unlike other adversarial examples, FIPatch demonstrates strong stealthiness and resistance to filters. To implement FIPatch in the physical world, it is essential to overcome the following challenges:
  - Challenge 1: How to accurately model fluorescent ink and determine the most effective attack parameters for FIPatch?
  - Challenge 2: How to enhance the robustness of FIPatch by leveraging the properties of fluorescent ink, making it more viable for real-world application?
- To address these challenges, we propose a four-module FIPatch attack framework as shown in this Figure.
  - The **Automatic Traffic Sign Localization** module automatically detects the valid region on a traffic sign for adding perturbations. The **Fluorescence Modeling** module simulates the application of fluorescent ink by adding colored circles with varying parameters to the identified region, replicating the perturbation effects. The **Fluorescence Optimization** module optimizes these parameters using goal-based and patch-aware loss functions and employs a particle swarm optimization algorithm to identify the most effective attack configuration. These three modules collectively address Challenge 1.
  - To tackle Challenge 2, the **Robustness Improvement** module customizes multiple transformation distributions to enhance the real-world robustness of FIPatch.

![image-20241217020457310](https://p.ipic.vip/tljejo.png)

