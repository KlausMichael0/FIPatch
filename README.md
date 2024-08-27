### ITPATCH: An Invisible and Triggered Physical Adversarial Patch against Traffic Sign Recognition

This is the official implementation in the paper **ITPATCH: An Invisible and Triggered Physical Adversarial Patch against Traffic Sign Recognition** in Python.



### Quickstart

1. Install the packages used in our scheme

   `pip install -r ./requirement.txt`

2. Set the following parameters in `itpatch_attack.py`:

   ```python
   dataset: datasets for model training and adversary attacks.
   				'ctsrd'; 'gtsrb'
   model_name: black-box model.
   				'resnet50'; 'resnet101'; 'vgg13'; 'vgg16' —— in ctsrd dataset.
     			'cnn'; 'inceptionv3'; 'mobilenetv2'; 'googlenet' —— in gtsrb dataset.
   perturb_radius: radius of the added perturbation.
   circle_number: number of fluorescent circles.
   n_restarts: n-random-restarts strategy for the particle swarm optimization.
   location: whether to record the perturbation center of a successful attack.
   color_test: whether to fix the perturbation color.
   color_used: if color_test is set to True, set the perturbation color
   save_folder: record images of failed attacks
   ```

3. Run the `itpatch_attack.py`

   `python itpatch_attack.py`

### Demonstration videos

- [ITPatch](https://sites.google.com/view/itpatch-attack/home)


### What is the framework?

- In this work, we present ITPatch, a physical adversarial example triggered by UV light with fluorescent effects. Unlike other adversarial patches, SPAE is able to be actively triggered by the attacker and is invisible when the attack is not triggered. In order to create ITPatch, it is important to address the following challenges: 

  - Challenge 1: How to model fluorescent ink and find the most effective attack parameters for ITPATCH?
  - Challenge 2: How to improve the robustness of ITPATCH based on the characteristics of fluorescent ink to make it more practical in the real world?
- To address these challenges, we designed a four-module SPAE attack as shown in this Figure. The **Automatic Traffic Sign Localization** module first identifies the legal region to add the perturbation. Next, to simulate the fluorescent ink applied to the traffic sign, the **Fluorescence Modeling** module adds colored circles with different parameters to the valid region. This module simulates the perturbation effect of the fluorescent material. The **Fluorescence Optimization** module optimizes the parameters of the fluorescent material based on a customized loss function and uses a particle swarm optimization algorithm to find the most efficient combination of attack parameters. The **Robustness Improvement** module improves the robustness of SPAE in the real world by customizing multiple transformation distributions.

![image-20240827150653810](https://s2.loli.net/2024/08/27/YFm1hZ3KboQATWN.png)

