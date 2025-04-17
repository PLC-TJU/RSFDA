# Riemannian Spatial Filtering and Domain Adaptation (RSFDA)
 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

[中文版本](./README.ch.md)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
- [Related Research Resources](#-related-research-resources)
- [Data Availability](#-data-availability)
- [Citation](#-citation)
- [Contact](#-contact)
- [License and Attribution](#license-and-attribution)

## Introduction

**RSFDA** 

The Riemannian Spatial Filtering with Domain Adaptation (RSFDA) algorithm is a pioneering framework that integrates Riemannian spatial filtering (RSF), manifold embedded feature alignment (MEFA), and stacking ensemble learning with time-frequency feature fusion (TF-Stacking). This synergy enables significant enhancement of cross-session motor imagery (MI) EEG classification performance without requiring extensive retraining. The core innovations lie in:

* Riemannian Spatial Filtering (RSF): Leveraging geometric optimization in Riemannian space to extract low-dimensional discriminative features, RSF dramatically reduces computational complexity while preserving critical neural information.
* Manifold Embedded Feature Alignment (MEFA): By aligning feature distributions across sessions through manifold learning, MEFA minimizes domain shifts and enhances generalization across temporal variations, eliminating the need for frequent model recalibration.
* TF-Stacking Ensemble Learning: Combining time-frequency domain feature fusion with hierarchical ensemble learning, this module amplifies classification accuracy and robustness, capturing both temporal and spectral discriminative patterns.

The algorithm’s modular architecture ensures seamless integration of these components, facilitating future extensions and adaptability. Collectively, RSFDA achieves superior efficiency, generalization, and precision, positioning it as a transformative solution for real-world BCI applications.

**Key Features**: Brain-computer interface ([BCI](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface)), motor imagery ([MI](https://en.wikipedia.org/wiki/Motor_imagery)), electroencephalography ([EEG](https://en.wikipedia.org/wiki/Electroencephalography)), [Riemannian geometry](https://en.wikipedia.org/wiki/Riemannian_geometry), [domain adaptation](https://en.wikipedia.org/wiki/Domain_adaptation) and cross-session.

## 📁 Project Structure
```plaintext
CTSSP/
├── requirements.txt        # Required packages
├── main_cross_session.py  # Cross-session classification pipeline

```

## 🔧 Installation & Setup

To install and run the project, please follow these steps:

1. Clone the repository locally
```bash
git clone https://github.com/PLC-TJU/RSFDA.git
cd RSFDA
```

2. Install the necessary dependencies
```bash
pip install -r requirements.txt
```

3. Install NeuroDecKit Toolbox
```bash
git clone https://github.com/PLC-TJU/NeuroDecKit.git
cd NeuroDecKit
python setup.py install
```

## 🚀 Quick Start

### Cross-Session Classification Example
```bash
python main_cross_session.py 
```

## 📚 Related Research Resources

We express our gratitude to the open-source community, which facilitates the broader dissemination of research by other researchers and ourselves. The coding style in this repository is relatively rough. We welcome anyone to refactor it to make it more efficient. Our model codebase is largely based on the following repositories:

- [<img src="https://img.shields.io/badge/GitHub-NeuroDeckit-b31b1b"></img>](https://github.com/PLC-TJU/NeuroDeckit) A Python toolbox for EEG signal processing and BCI applications. It includes various preprocessing methods, feature extraction techniques, and classification algorithms.
- [<img src="https://img.shields.io/badge/GitHub-RSF-b31b1b"></img>](https://github.com/PLC-TJU/RSF) Riemannian geometry-based spatial filtering (RSF) is a method based on Riemannian geometry designed to improve the accuracy of MI EEG signal classification.
- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb) An open science project aimed at establishing a comprehensive benchmark for BCI algorithms using widely available EEG datasets.
- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode) Contains several deep learning models such as EEGNet, ShallowConvNet, and DeepConvNet, designed specifically for EEG signal classification. Braindecode aims to provide an easy-to-use deep learning toolbox.
- [<img src="https://img.shields.io/badge/GitHub-CSPNet-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet) Contains Tensor-CSPNet and Graph-CSPNet, two deep learning models for MI-EEG signal classification.
- [<img src="https://img.shields.io/badge/GitHub-LMDANet-b31b1b"></img>](https://github.com/MiaoZhengQing/LMDA-Code) A deep learning-based network for EEG signal classification. LMDA-Net combines various advanced neural network architectures to enhance classification accuracy.

## 📊 Data Availability

We used the following public datasets:

**Table 1** Details of all public datasets

| Dataset                                                   |     Classes     | Sessions | Trials | Channels | Duration (s) | Subjects |
| :-------------------------------------------------------  | :-------------: | :------: | :----: | :------: | :----------: | :------: |
| [Pan2023](https://doi.org/10.1088/1741-2552/ad0a01)       | left/right hand |    2     |  240   |    28    |      4       |    14    |
| [BNCI001-2014](https://doi.org/10.3389/fnins.2012.00055)  | left/right hand |    2     |  288   |    22    |      4       |    9     |
| [BNCI001-2015](https://doi.org/10.1109/TNSRE.2012.2189584)| right hand/feet |    2     |  400   |    13    |      5       |    12    |
| **Total:**                                                |                 |          |        |          |              |  **35**  |


## 📜 Citation
If you use this code, please cite:  
```
@article{pan2025rsfda,
  title={Cross-session motor imagery-electroencephalography decoding with Riemannian spatial filtering and domain adaptation}, 
  author={Lincong, Pan and Xinwei, Sun and Kun, Wang and Yupei, Cao and Minpeng, Xu and Dong, Ming},
  journal={Journal of Biomedical Engineering},
  year={2025},
  month={April},
  volume={42},
  number={2},
}
```

## 🤝 Contact

If you have any questions or concerns, please contact us at:  
 - Email: panlincong@tju.edu.cn

## 📝 License and Attribution

© 2024 Lincong Pan. MIT License.  
Please refer to the [LICENSE](./LICENSE) file for details on the licensing of our code.   
 
