# 黎曼空间滤波与域适应算法 (RSFDA)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

[English Version](./README.md)

## 目录

- [项目简介](#项目简介)
- [项目结构](#-项目结构)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [相关研究资源](#-相关研究资源)
- [数据可用性](#-数据可用性)
- [引用方式](#-引用方式)
- [联系我们](#-联系我们)
- [许可协议](#-许可协议)

## 项目简介

**RSFDA**  

黎曼空间滤波与域适应算法（RSFDA）是一个创新性框架，集成了以下三大核心技术：
1. **黎曼空间滤波（RSF）**：在黎曼流形空间中进行几何优化，提取低维判别特征，在保留关键神经信息的同时显著降低计算复杂度
2. **流形嵌入特征对齐（MEFA）**：通过流形学习方法对齐跨会话的特征分布，最小化域偏移，增强模型对时间变化的泛化能力
3. **时频特征融合的堆叠集成学习（TF-Stacking）**：结合时频域特征融合与分层集成学习，提升分类精度和鲁棒性

该算法的模块化架构支持各组件的无缝集成，便于未来扩展。RSFDA在计算效率、泛化能力和分类精度方面表现卓越，为实际BCI应用提供了突破性解决方案。

**核心技术**：脑机接口（[BCI](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface)）、运动想象（[MI](https://en.wikipedia.org/wiki/Motor_imagery)）、脑电图（[EEG](https://en.wikipedia.org/wiki/Electroencephalography)）、[黎曼几何](https://en.wikipedia.org/wiki/Riemannian_geometry)、[域适应](https://en.wikipedia.org/wiki/Domain_adaptation)和跨会话分类。

## 📁 项目结构
```plaintext
RSFDA/
├── requirements.txt        # 依赖包清单
├── main_cross_session.py   # 跨会话分类主程序
|
├── matlab version/         # matlab版本的源代码
|   ├── rsfda/              # RSFDA算法源代码
|   ├── rsfda_modeling      # RSFDA模型训练代码
|   ├── rsfda_classify      # RSFDA模型分类代码        
```

## 🔧 安装指南

安装和运行该项目，请按照以下步骤进行：

1. 克隆项目到本地
```bash
git clone https://github.com/PLC-TJU/RSFDA.git
cd RSFDA
```
2. 安装所需依赖包
```bash
pip install -r requirements.txt
```

3. 安装NeuroDecKit工具箱  
```bash
git clone https://github.com/PLC-TJU/NeuroDecKit.git
cd NeuroDecKit
python setup.py install
```

## 🚀 快速开始

### 跨时间分类示例
```bash
python main_cross_session.py 
```

## 📚 相关研究资源

本项目的实现基于以下开源项目，特此致谢：

- [<img src="https://img.shields.io/badge/GitHub-NeuroDeckit-b31b1b"></img>](https://github.com/PLC-TJU/NeuroDeckit) 
EEG信号全流程处理工具箱
- [<img src="https://img.shields.io/badge/GitHub-RSF-b31b1b"></img>](https://github.com/PLC-TJU/RSF)
基于黎曼几何的空间滤波算法
- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb)
BCI算法的开源基准测试框架
- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode)
面向EEG信号的深度学习工具箱，包括EEGNet、ShallowConvNet和DeepConvNet等多种模型
- [<img src="https://img.shields.io/badge/GitHub-CSPNet-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet)
基于张量和图结构的CSP神经网络模型，包括Tensor-CSPNet和Graph-CSPNet
- [<img src="https://img.shields.io/badge/GitHub-LMDANet-b31b1b"></img>](https://github.com/MiaoZhengQing/LMDA-Code)
集成多维度注意力模块的轻量级神经网络模型

## 📊 数据可用性

使用的公开数据集信息如下：

**表 1** 所有公开数据集的详细信息

| 数据集名称                                                 |  类别  | 会话数 | 试次数 | 通道数 | 持续时间（秒） | 被试数 |
| :-------------------------------------------------------  | :----: | :----: | :----: | :----: | :----------: | :------: |
| [Pan2023](https://doi.org/10.1088/1741-2552/ad0a01)       | 左/右手 |    2     |  240   |    28    |      4       |    14    |
| [BNCI001-2014](https://doi.org/10.3389/fnins.2012.00055)  | 左/右手 |    2     |  288   |    22    |      4       |    9     |
| [BNCI001-2015](https://doi.org/10.1109/TNSRE.2012.2189584)| 右手/脚 |    2     |  400   |    13    |      5       |    12    |
| **总计:**                                                 |         |          |        |          |              |  **35**  |


## 📜 引用方式
如果您使用了本代码，请至少引用以下一篇文章， 非常感谢您的支持：  

```bibtex
@article{pan2025rsfda,
  title={基于黎曼空间滤波与域适应的跨时间运动想象-脑电解码研究}, 
  author={潘林聪, 孙新维, 王坤, 曹愉培, 许敏鹏, 明东},
  journal={生物医学工程学杂志},
  month={4},
  year={2025},
  volume={42},
  number={2},
  pages={272-279},
  doi={10.7507/1001-5515.202411035},
  issn={1001-5515},
}
```
```bibtex
@article{pan2025rsf,
  title={Enhancing Motor Imagery EEG Classification with a Riemannian Geometry-Based Spatial Filtering (RSF) Method}, 
  author={Lincong, Pan and Kun, Wang and Yongzhi Huang and Xinwei, Sun and Jiayuan Meng and Weibo Yi and Minpeng, Xu and Tzyy-Ping Jung and Dong, Ming},
  journal={Neural Networks},
  year={2025},
  volume={188},
  pages={107511},
  doi={10.1016/j.neunet.2025.107511},
  publisher={Elsevier}
}
```
```bibtex
@article{pan2023rave,
  title={Riemannian geometric and ensemble learning for decoding cross-session motor imagery electroencephalography signals}, 
  author={Lincong, Pan and Kun, Wang and Lichao Xu and Xinwei, Sun and Weibo Yi and Minpeng, Xu and Dong, Ming},
  journal={Journal of Neural Engineering},
  year={2023},
  volume={20},
  number={6},
  pages={066011},
  doi={10.1088/1741-2552/ad0a01},
  publisher={IOP Publishing}
}
```

## 🤝 联系我们

如果您有任何问题或疑问，请联系我们：  
 - 邮箱：panlincong@tju.edu.cn

## 📝 许可协议

© 2024 潘林聪. MIT许可证
详见[LICENSE](./LICENSE)文件
