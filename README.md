# mri-deep-learning-tools
Open-source libraries for MRI images processing and deep learning. 
Updates in 2020: `monai`,`torchio`, `medicalzooputorch`, `transunet` virused.
Updates in 2021:  `torchio` wrote an adds-in for `monai`. There is **super-useful** tutorials from `MONAI` and `NVIDIA` for almost all tasks that you need in MRI imaging in 2D and 3D. [Check this out](https://github.com/Project-MONAI/tutorials) for `pytorch` and `tensorflow`. Also check [this](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/nnunet_for_pytorch) catalog for containerized solutions. `DALI` augmentations just saving lots of time!

## Pytorch

| Project Name | Description | Scenario |
| ------- | ------ | ------ |
| [MONAI](https://github.com/Project-MONAI/MONAI) | Medical Open Network for AI — PyTorch-based framework for deep learning in healthcare imaging, part of PyTorch Ecosystem. | preprocessing, classification, segmentation |
| [SegNet](https://github.com/alexgkendall/caffe-segnet) | Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. [Paper](http://arxiv.org/abs/1511.00561) | segmentation |
| [Medical Detection Toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit) | 2D + 3D implementations of Mask R-CNN, Retina Net, Retina U-Net for medical images. | detection, segmentation |
| [TorchIO](https://github.com/fepegar/torchio) | Tools to read, preprocess, sample, augment, and write 3D medical images in PyTorch. [Paper](https://arxiv.org/abs/2003.04696) | training, preprocessing |
| [DeepMedic](https://github.com/deepmedic/deepmedic) | Multi-Scale 3D CNN for Segmentation of 3D Medical Scans. Processes NIFTI images. | segmentation |
| [MedicalTorch](https://github.com/perone/medicaltorch) | PyTorch framework with loaders, pre-processors and datasets for medical imaging. | preprocessing |
| [MEDICAL ZOO](https://github.com/black0017/MedicalZooPytorch) | 3D multi-modal medical image segmentation library in PyTorch. | segmentation |
| [TransUNet](https://github.com/Beckschen/TransUNet) | Transformers (ViT) for medical image segmentation. | segmentation |
| [nipy ecosystem](https://nipy.org/) | Community for Python neuroimaging: [dipy](https://github.com/dipy/dipy) (diffusion), [nibabel](https://github.com/nipy/nibabel) (I/O), [nilearn](https://github.com/nilearn/nilearn) (ML), [MNE](https://github.com/mne-tools/mne-python) (EEG/MEG) | denoising, registration, reconstruction, visualization |
| [Deep-MRI-Reconstruction](https://github.com/js3611/Deep-MRI-Reconstruction) | DC-CNN and CRNN-MRI for MR image reconstruction from undersampled measurements. | reconstruction |
| [SCT](https://github.com/neuropoly/spinalcordtoolbox) | Spinal Cord Toolbox — processing and analysis of spinal cord MRI data. | preprocessing, segmentation |
| [TractSeg](https://github.com/MIC-DKFZ/TractSeg) | Fast white matter bundle segmentation from Diffusion MRI with tractography. | segmentation |
| [clinicadl](https://github.com/aramis-lab/clinicadl) | Reproducible CNN experiments for Alzheimer’s disease classification from MRI. | classification |
| [BrainPrep](https://github.com/quqixun/BrainPrep) | Brain MRI preprocessing pipeline. | preprocessing |
| [GANCS](https://github.com/gongenhao/GANCS) | Compressed Sensing MRI using GANs. | reconstruction |
| [Pytorch-LRP](https://github.com/moboehle/Pytorch-LRP) | Layer-wise relevance propagation for MRI-based Alzheimer’s classification. [Paper](https://arxiv.org/abs/1903.07317) | classification |
| [3D-UNet-PyTorch](https://github.com/JielongZ/3D-UNet-PyTorch-Implementation) | 3D UNet implementation. [Paper](https://arxiv.org/abs/1606.06650) | segmentation |
| [NiftyTorch](https://github.com/NiftyTorch/NiftyTorch.v.0.1) | Python API for deploying DNNs in neuroimaging research. | classification, segmentation |
| [DeepPipe](https://github.com/neuro-ml/deep_pipe) | Medical image manipulation — parallel training, optimization, utils. | model training |
| [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit) | Docker images for BRATS tumor segmentation solutions. | segmentation |




## TensorFlow

| Project Name | Description | Scenario |
| ------- | ------ | ------ |
| [NiftyNet](https://github.com/NifTK/NiftyNet) | TensorFlow-based CNN platform for medical image analysis and image-guided therapy. | classification, segmentation |
| [DLTK](https://github.com/DLTK/DLTK) | Deep Learning Toolkit for Medical Imaging — fast prototyping with focus on reproducibility. | classification, segmentation, super-resolution |
| [CNN-3D-images-Tensorflow](https://github.com/jibikbam/CNN-3D-images-Tensorflow) | MRI classification using 3D CNN. | classification |
| [population-gcn](https://github.com/parisots/population-gcn) | Graph CNNs for semi-supervised disease prediction using population graphs (ABIDE dataset). | classification |
| [Nobrainer](https://github.com/neuronets/nobrainer) | DL framework for 3D image processing with TensorFlow/Keras. | preprocessing, segmentation |
| [3D-Convnet-Alzheimers](https://github.com/RishalAggarwal/3D-Convnet-for-Alzheimer-s-Detection) | Alzheimer's detection from T1 MRI scans (ADNI database). | detection, preprocessing |
