# mri-deep-learning-tools

> Curated list of open-source deep learning tools for MRI and neuroimaging.

## What's New (2024-2026)

- **Brain Foundation Models**: BrainLM (6700h fMRI), BrainSegFounder, BrainMorph
- **MedSAM2** (ICLR 2025): Promptable 3D segmentation for medical images & videos
- **nnWNet** (CVPR 2025): Transformer benchmark for biomedical segmentation
- **ACTION Toolbox**: Pretrained models on 3800+ fMRI scans

## Contents

- [Foundation Models](#foundation-models)
- [Getting Started](#getting-started)
- [Preprocessing & Pipelines](#preprocessing--pipelines)
- [Segmentation](#segmentation)
- [Classification & Prognosis](#classification--prognosis)
- [Reconstruction](#reconstruction)
- [Diffusion & Tractography](#diffusion--tractography)
- [Visualization & I/O](#visualization--io)
- [Contributing](#contributing)

---

## Foundation Models

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [fmri-fm](https://github.com/MedARC-AI/fmri-fm) | ![Stars](https://img.shields.io/github/stars/MedARC-AI/fmri-fm?style=flat-square) | MedARC fMRI foundation model | - | PyTorch | 2024 |
| [BrainSegFounder](https://github.com/lab-smile/BrainSegFounder) | ![Stars](https://img.shields.io/github/stars/lab-smile/BrainSegFounder?style=flat-square) | Brain segmentation foundation model | - | PyTorch | 2024 |
| [BrainMorph](https://github.com/alanqrwang/brainmorph) | ![Stars](https://img.shields.io/github/stars/alanqrwang/brainmorph?style=flat-square) | Foundational keypoint model for brain MRI registration | - | PyTorch | 2024 |
| [ACTION](https://github.com/mxliu/ACTION-Software-for-Functional-MRI-Analysis) | ![Stars](https://img.shields.io/github/stars/mxliu/ACTION-Software-for-Functional-MRI-Analysis?style=flat-square) | Toolbox with pretrained models on 3800+ fMRI scans | [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811924004646) | PyTorch | 2024 |
| [BrainLM](https://github.com/vandijklab/BrainLM) | ![Stars](https://img.shields.io/github/stars/vandijklab/BrainLM?style=flat-square) | Foundation model for fMRI, trained on 6700h of recordings | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1) | PyTorch | 2023 |

## Getting Started

Essential tools for MRI deep learning workflows.

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [TorchIO](https://github.com/fepegar/torchio) | ![Stars](https://img.shields.io/github/stars/fepegar/torchio?style=flat-square) | Tools to read, preprocess, augment 3D medical images | [arXiv](https://arxiv.org/abs/2003.04696) | PyTorch | 2020 |
| [MONAI](https://github.com/Project-MONAI/MONAI) | ![Stars](https://img.shields.io/github/stars/Project-MONAI/MONAI?style=flat-square) | Medical Open Network for AI — PyTorch framework for healthcare imaging | [Docs](https://docs.monai.io/) | PyTorch | 2019 |
| [nilearn](https://github.com/nilearn/nilearn) | ![Stars](https://img.shields.io/github/stars/nilearn/nilearn?style=flat-square) | Statistical learning on neuroimaging data | [Docs](https://nilearn.github.io/) | Python | 2014 |
| [nibabel](https://github.com/nipy/nibabel) | ![Stars](https://img.shields.io/github/stars/nipy/nibabel?style=flat-square) | Read/write neuroimaging file formats (NIfTI, GIFTI, etc.) | [Docs](https://nipy.org/nibabel/) | Python | 2006 |

## Preprocessing & Pipelines

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [BrainPrep](https://github.com/quqixun/BrainPrep) | ![Stars](https://img.shields.io/github/stars/quqixun/BrainPrep?style=flat-square) | Brain MRI preprocessing pipeline | - | Python | 2021 |
| [DeepPipe](https://github.com/neuro-ml/deep_pipe) | ![Stars](https://img.shields.io/github/stars/neuro-ml/deep_pipe?style=flat-square) | Medical image manipulation — parallel training, optimization | - | PyTorch | 2019 |
| [Nobrainer](https://github.com/neuronets/nobrainer) | ![Stars](https://img.shields.io/github/stars/neuronets/nobrainer?style=flat-square) | DL framework for 3D image processing | [Docs](https://github.com/neuronets/nobrainer) | TensorFlow | 2019 |
| [MedicalTorch](https://github.com/perone/medicaltorch) | ![Stars](https://img.shields.io/github/stars/perone/medicaltorch?style=flat-square) | Loaders, pre-processors and datasets for medical imaging | - | PyTorch | 2018 |
| [SCT](https://github.com/neuropoly/spinalcordtoolbox) | ![Stars](https://img.shields.io/github/stars/neuropoly/spinalcordtoolbox?style=flat-square) | Spinal Cord Toolbox for spinal cord MRI analysis | [Docs](https://spinalcordtoolbox.com/) | Python | 2014 |

## Segmentation

### SAM Family (Segment Anything)

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [MedSAM2](https://github.com/bowang-lab/MedSAM2) | ![Stars](https://img.shields.io/github/stars/bowang-lab/MedSAM2?style=flat-square) | 3D medical images & videos segmentation (ICLR 2025) | [Project](https://medsam2.github.io/) | PyTorch | 2025 |
| [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) | ![Stars](https://img.shields.io/github/stars/Joey-S-Liu/MedSAM3?style=flat-square) | SAM with medical concepts | - | PyTorch | 2025 |
| [MedSAM](https://github.com/bowang-lab/MedSAM) | ![Stars](https://img.shields.io/github/stars/bowang-lab/MedSAM?style=flat-square) | SAM for medical images, 1.5M image-mask pairs | [Nature Comms](https://www.nature.com/articles/s41467-024-44824-z) | PyTorch | 2024 |
| [MedLSAM](https://github.com/openmedlab/MedLSAM) | ![Stars](https://img.shields.io/github/stars/openmedlab/MedLSAM?style=flat-square) | Localize and Segment Anything for 3D medical images | - | PyTorch | 2024 |

### nnUNet Family

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [nnWNet](https://github.com/Yanfeng-Zhou/nnWNet) | ![Stars](https://img.shields.io/github/stars/Yanfeng-Zhou/nnWNet?style=flat-square) | CVPR 2025 transformer benchmark for biomedical segmentation | - | PyTorch | 2025 |
| [nnUNet](https://github.com/MIC-DKFZ/nnUNet) | ![Stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet?style=flat-square) | Self-configuring segmentation — gold standard | [Nature Methods](https://www.nature.com/articles/s41592-020-01008-z) | PyTorch | 2018 |

### Other Segmentation

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [TransUNet](https://github.com/Beckschen/TransUNet) | ![Stars](https://img.shields.io/github/stars/Beckschen/TransUNet?style=flat-square) | Transformers (ViT) for medical image segmentation | [arXiv](https://arxiv.org/abs/2102.04306) | PyTorch | 2021 |
| [MEDICAL ZOO](https://github.com/black0017/MedicalZooPytorch) | ![Stars](https://img.shields.io/github/stars/black0017/MedicalZooPytorch?style=flat-square) | 3D multi-modal medical image segmentation library | - | PyTorch | 2020 |
| [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit) | ![Stars](https://img.shields.io/github/stars/neuronflow/BraTS-Toolkit?style=flat-square) | Docker images for BraTS tumor segmentation | - | Docker | 2020 |
| [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit) | ![Stars](https://img.shields.io/github/stars/MIC-DKFZ/medicaldetectiontoolkit?style=flat-square) | Mask R-CNN, Retina Net, Retina U-Net for medical images | - | PyTorch | 2018 |
| [NiftyNet](https://github.com/NifTK/NiftyNet) | ![Stars](https://img.shields.io/github/stars/NifTK/NiftyNet?style=flat-square) | CNN platform for medical image analysis | [Paper](https://www.sciencedirect.com/science/article/pii/S0169260717311823) | TensorFlow | 2017 |
| [DeepMedic](https://github.com/deepmedic/deepmedic) | ![Stars](https://img.shields.io/github/stars/deepmedic/deepmedic?style=flat-square) | Multi-Scale 3D CNN for medical scan segmentation | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841516301839) | TensorFlow | 2016 |
| [3D-UNet-PyTorch](https://github.com/JielongZ/3D-UNet-PyTorch-Implementation) | ![Stars](https://img.shields.io/github/stars/JielongZ/3D-UNet-PyTorch-Implementation?style=flat-square) | 3D UNet implementation | [arXiv](https://arxiv.org/abs/1606.06650) | PyTorch | 2016 |
| [SegNet](https://github.com/alexgkendall/caffe-segnet) | ![Stars](https://img.shields.io/github/stars/alexgkendall/caffe-segnet?style=flat-square) | Encoder-Decoder architecture for segmentation | [arXiv](http://arxiv.org/abs/1511.00561) | Caffe | 2015 |

## Classification & Prognosis

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [clinicadl](https://github.com/aramis-lab/clinicadl) | ![Stars](https://img.shields.io/github/stars/aramis-lab/clinicadl?style=flat-square) | Reproducible CNN experiments for Alzheimer's classification | [Docs](https://clinicadl.readthedocs.io/) | PyTorch | 2020 |
| [NiftyTorch](https://github.com/NiftyTorch/NiftyTorch.v.0.1) | ![Stars](https://img.shields.io/github/stars/NiftyTorch/NiftyTorch.v.0.1?style=flat-square) | Python API for deploying DNNs in neuroimaging | - | PyTorch | 2020 |
| [Pytorch-LRP](https://github.com/moboehle/Pytorch-LRP) | ![Stars](https://img.shields.io/github/stars/moboehle/Pytorch-LRP?style=flat-square) | Layer-wise relevance propagation for MRI classification | [arXiv](https://arxiv.org/abs/1903.07317) | PyTorch | 2019 |
| [3D-Convnet-Alzheimers](https://github.com/RishalAggarwal/3D-Convnet-for-Alzheimer-s-Detection) | ![Stars](https://img.shields.io/github/stars/RishalAggarwal/3D-Convnet-for-Alzheimer-s-Detection?style=flat-square) | Alzheimer's detection from T1 MRI (ADNI database) | - | TensorFlow | 2019 |
| [CNN-3D-images-Tensorflow](https://github.com/jibikbam/CNN-3D-images-Tensorflow) | ![Stars](https://img.shields.io/github/stars/jibikbam/CNN-3D-images-Tensorflow?style=flat-square) | MRI classification using 3D CNN | - | TensorFlow | 2018 |
| [DLTK](https://github.com/DLTK/DLTK) | ![Stars](https://img.shields.io/github/stars/DLTK/DLTK?style=flat-square) | Deep Learning Toolkit for Medical Imaging | [Paper](https://arxiv.org/abs/1711.06853) | TensorFlow | 2017 |
| [population-gcn](https://github.com/parisots/population-gcn) | ![Stars](https://img.shields.io/github/stars/parisots/population-gcn?style=flat-square) | Graph CNNs for disease prediction (ABIDE dataset) | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_21) | TensorFlow | 2017 |

## Reconstruction

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [GANCS](https://github.com/gongenhao/GANCS) | ![Stars](https://img.shields.io/github/stars/gongenhao/GANCS?style=flat-square) | Compressed Sensing MRI using GANs | - | PyTorch | 2018 |
| [Deep-MRI-Reconstruction](https://github.com/js3611/Deep-MRI-Reconstruction) | ![Stars](https://img.shields.io/github/stars/js3611/Deep-MRI-Reconstruction?style=flat-square) | DC-CNN and CRNN-MRI for undersampled MR reconstruction | [Paper](https://ieeexplore.ieee.org/document/8067520) | PyTorch | 2017 |

## Diffusion & Tractography

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [TractSeg](https://github.com/MIC-DKFZ/TractSeg) | ![Stars](https://img.shields.io/github/stars/MIC-DKFZ/TractSeg?style=flat-square) | White matter bundle segmentation from diffusion MRI | [Paper](https://www.sciencedirect.com/science/article/pii/S1053811918306864) | PyTorch | 2018 |
| [dipy](https://github.com/dipy/dipy) | ![Stars](https://img.shields.io/github/stars/dipy/dipy?style=flat-square) | Diffusion MRI analysis — tractography, denoising, registration | [Docs](https://dipy.org/) | Python | 2014 |

## Visualization & I/O

| Tool | Stars | Description | Paper | Framework | Year |
|------|-------|-------------|-------|-----------|------|
| [MNE-Python](https://github.com/mne-tools/mne-python) | ![Stars](https://img.shields.io/github/stars/mne-tools/mne-python?style=flat-square) | MEG and EEG data processing | [Docs](https://mne.tools/) | Python | 2013 |
| [nipy](https://github.com/nipy/nipy) | ![Stars](https://img.shields.io/github/stars/nipy/nipy?style=flat-square) | Neuroimaging analysis in Python | [Docs](https://nipy.org/nipy/) | Python | 2009 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding tools.

**Quick format:**
```
| [ToolName](url) | ![Stars](https://img.shields.io/github/stars/owner/repo?style=flat-square) | Description | [Paper](url) | Framework | Year |
```

## License

[CC-BY-4.0](LICENSE)
