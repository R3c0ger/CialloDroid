<!--suppress ALL -->

<div align="center">
    <!--<img src="resources/logo/logo_256.png" alt="logo_256" style="height: 120px" /> -->
	<h1 style="padding-bottom: .3em !important; border-bottom: 1.5px solid #d0d7deb3 !important;">CialloDroid</h1>
</div>

<h2 align="center" style="border-bottom-style: none !important;"> 信息安全创新综合实验——基于图神经网络的安卓恶意软件检测模型 </h2>

<p align="center" style="text-align:center">Ciallo ～(∠・ω< )⌒★</p>

<p align="center" style="text-align:center">
    <img src="https://img.shields.io/badge/version-v0.1.0-brightgreen" alt="version">
    <img src="https://img.shields.io/badge/python-3.9+-yellow" alt="python">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="contributions">
</p>

<p  align="center" style="text-align:center"> 中文 | <a href="README.md"> English </a> </p>

## 任务书

### 任务简介

安卓操作系统作为移动设备上最广泛使用的开放源代码操作系统，具有着丰富的应用市场与客户人群，但其普及率与快速的增长也为恶意软件发展提供了温床。

为此，现有研究者通过提取程序特征来实现恶意软件检测机制。但是这些方法往往 **忽视了程序语义信息**，导致恶意软件检测精度不理想。

为此，该系统设计题目需要针对安卓应用软件，**引入图卷积神经网络**，能够将程序语义关系转换为图模型，进而采用图分析方法，实现精确度高、检测速度快的恶意软件检测方法。

### 相关工作

现有基于深度学习的安卓恶意软件检测方法主要可以分为 **基于语法特征** 和 **基于语义特征** 的检测方法。

基于语法特征的方法主要关注于代码的语法结构，通过从语法结构中提取出代表性的特征与恶意行为标签，来训练模型从而实施检测。该方法的优点在于可以快速准确地识别简单和常见的恶意代码，但是对于复杂和高级的恶意软件则表现出相对较差的准确性和效率。

基于语义特征的方法通常通常会采用程序分析技术。程序分析是一种将程序的行为抽象为数学模型的技术，以便进一步展示和分析，相比于基于语法特征的方法，该方法可以更好地识别复杂和高级的恶意软件，以及从中提取有用的信息。通过程序分析，研究者可以更好地提取恶意软件执行过程中的语义信息，以理解其行为和特征，这类特征往往会通过图和数据流的方法进行表达。

### 可行方法

该系统可行的方法是对安卓应用程序首先进行静态分析，提取应用程序的函数调用图，并利用现有 [敏感 API 数据集](https://apichecker.github.io/) 对函数调用图进行自动标注。然后设计图卷积神经网络模型，并通过 One-Hot 编码实现对函数调用图的特征编码，对图卷积神经网络模型每个节点特征向量进行初始化，最后利用现有数据集对模型进行训练。

其中安卓应用程序静态分析工具可以采用 [Androguard](https://github.com/androguard/androguard)。恶意安卓应用可以从 [VirusShare](https://virusshare.com/) 平台上下载，正常安卓应用可以从 [AndroZoo](https://androzoo.uni.lu/) 平台上下载。

![architecture](img/architecture-zh.svg)

### 任务要求

1. 查阅相关资料，了解恶意软件检测的原理；
2. 设计实现安卓应用程序的函数调用图提取及其敏感 API 调用自动标注方法；
3. 设计实现基于图卷积神经网络模型的安卓恶意软件模型训练方法；
4. 设计实验分析方法，要求测试恶意应用和正常应用均不少于 50 个，评估模型的准确率、召回率和 F1 得分；
5. 根据小组分工，撰写个人课程报告，形成整体作品报告。

## 致谢

- [Androguard](https://github.com/androguard/androguard)
- [VirusShare](https://virusshare.com/)
- [AndroZoo](https://androzoo.uni.lu/)

## 许可证

本项目遵循 Apache 2.0 许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。