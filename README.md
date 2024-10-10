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
In this work, we propose to generate realistic paired data on unlabeled image modalities (OCT-A and FA) with annotated source domain (fundus photography). The data generation is achieved by conditional semantic diffusion. The two important takeaways are:
>- The conditional diffusion can be trained with weak supervision. And we found the feasible conditions for the noisy label.
>- The synthetic paired data can be used to train/finetune segmentation model on the unlabeled target domains.

<p align="center">
  <img src="/assets/workflow.png" alt="drawing" width="650"/>
</p>

The workflow include (1) training a segmentation network with the annotated source domain data and create pseudo-labels of the target domain data, (2) training the conditional diffusion model with the pseudo-labels, (3) generate synthetic target domain images with real labels and (4) finetune the segmentation model trained with source domain data. Since the step (1) and (4) are trivial, we focus on step (2) and (3) in this repository. However, the network architechture aplied for the segmentation model is also included. A brief description of the code:

* `diffusion_solver.py`: This is the major code to define the diffusion process (setting variance schedule), conduct forward and reverse process. In the class **_DiffusionSampler_**, we define methods **forward_sample** and **reverse_sample**. The reverse sampling can be either conditional or unconditional. For better visualization, there are some auxiliary methods, e.g., iteratively plot each denoising step (**serial_reverse_iterate**).

* `conditional_diffusion_trainer.py`: This is the training code for the diffusion model in step (2). Note that a pre-trained segmentation model is needed to provide the pseudo-label. In the release, we provide a model checkpoint [diffusion.octa500.pt](https://github.com/DeweiHu/AdaptDiff/releases/tag/octa-500-v1.0) for generating OCT-A images in the same style with the [OCTA-500](https://ieee-dataport.org/open-access/octa-500) dataset.      

* `conditional_diffusion_tester.py`: This is the inference code for the diffusion model in step (3). We save the synthetic paired data in a pickle file.

* `example_inference.py`: **TODO**, provide a simple inference code to show the generation result with a binary mask.

### Data arrangement


