![](https://img.shields.io/badge/Language-python-brightgreen.svg)

# Conditional Diffusion for Medical Image Synthesis
### [2024 MICCAI SASHIMI] AdaptDiff: Cross-Modality Domain Adaptation via Weak Conditional Semantic Diffusion for Retinal Vessel Segmentation
---

### Environment
The virtual environment is created by `Diffusion.yaml` by running: 
```
conda env create -f Diffusion.yaml
```

### Method
<p align="center">
  <img src="/assets/workflow.png" alt="drawing" width="650"/>
</p>

The workflow include (1) training a segmentation network with the annotated source domain data and create pseudo-labels of the target domain data, (2) training the conditional diffusion model with the pseudo-labels, (3) generate synthetic target domain images with real labels and (4) finetune the segmentation model trained with source domain data. Since the step (1) and (4) are trivial, we focus on step (2) and (3) in this repository. However, the network architechture aplied for the segmentation model is also included. A brief description of the code:

* `diffusion_solver.py`: This is the major code to define the diffusion process (setting variance schedule), conduct forward and reverse process. In the class **_DiffusionSampler_**, we define methods **forward_sample** and **reverse_sample**. The reverse sampling can be either conditional or unconditional. For better visualization, there are some auxiliary methods, e.g., iteratively plot each denoising step (**serial_reverse_iterate**).

* `conditional_diffusion_trainer.py`: This is the training code for the diffusion model in step (2). Note that a pre-trained segmentation model is needed to provide the pseudo-label.     

* `conditional_diffusion_tester.py`: This is the inference code for the diffusion model in step (3). We save the synthetic paired data in a pickle file.


### Data arrangement


