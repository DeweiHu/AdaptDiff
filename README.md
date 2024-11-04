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

* `adapter_inference.py`: This is an example code to inference the model provided. Simply specify the directory to a binary vessel mask, model checkpoint and result save directory in `run_inference.sh`. Note that our model can work with image shape in either (384, 384) or (256, 256). You can try the code with the binary_mask.png provided, simply run
```
bash run_inference.sh
```
The following images shows the synthetic OCTA500 image with a (384, 384) binary vessel mask cropped from the DRIVE fundus photography image.

<p align="center">
  <img src="/assets/original_img.png" width="20%" alt="Image 1">
  <img src="/assets/binary_mask.png" width="20%" alt="Image 2">
  <img src="/assets/test_result.png" width="20%" alt="Image 3">
</p>


### Data arrangement
Our experiment includes several public datasets. For easier access, we use a dictionary to save all the data. Key for image lists are formated as {dataset name}_im and the corresponding ground truth are {dataset name}_gt. 

* `dataloader.py`: This is the code that create the dataloader with the paired data. For training, we randomly crop patches with size [256, 256] on the images. The number of patches on each image in specified with num_sample.


### Citation
If you find this project useful, please consider giving a star and cite our work:
```
  @inproceedings{hu2024adaptdiff,
  title={AdaptDiff: Cross-Modality Domain Adaptation via Weak Conditional Semantic Diffusion for Retinal Vessel Segmentation},
  author={Hu, Dewei and Li, Hao and Liu, Han and Wang, Jiacheng and Yao, Xing and Lu, Daiwei and Oguz, Ipek},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  pages={13--23},
  year={2024},
  organization={Springer}
}
```

If you have any question, feel free to contact me at: <br>
hudewei1212@gmail.com <br>

