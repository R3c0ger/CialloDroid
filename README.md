<!--suppress ALL -->

<div align="center">
    <!--<img src="resources/logo/logo_256.png" alt="logo_256" style="height: 120px" /> -->
	<h1 style="padding-bottom: .3em !important; border-bottom: 1.5px solid #d0d7deb3 !important;">CialloDroid</h1>
</div>

<h2 align="center" style="border-bottom-style: none !important;">Comprehensively Innovative Experiment of Information Security - Android Malware Detection Model Based on Graph Neural Network</h2>

<p align="center" style="text-align:center">
    <img src="https://img.shields.io/badge/version-v0.1.0-brightgreen" alt="version">
    <img src="https://img.shields.io/badge/python-3.9+-yellow" alt="python">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="contributions">
</p>

<p style="text-align:center">English | <a href="README_zh-CN.md">中文</a></p>

## Task Description

### Task Overview

The Android operating system, as the most widely used open-source operating system on mobile devices, boasts a rich application market and a large user base. However, its popularity and rapid growth have also provided fertile ground for the development of malicious software.

To address this issue, current researchers extract program features to implement mechanisms for detecting malicious software. However, these methods often **neglect program semantic information**, leading to suboptimal detection accuracy.

Therefore, this project aims to target Android applications by **introducing graph convolutional neural networks (GCNs)**. The goal is to convert program semantic relationships into graph models and use graph analysis methods to achieve a high-precision and fast malicious software detection method.

### Related Work

Existing deep learning-based Android malware detection methods can primarily be categorized into **syntax-based feature** and **semantics-based feature** detection methods.

Syntax-based methods focus on the syntactic structure of the code, extracting representative features and malicious behavior labels from the syntactic structure to train models for detection. The advantage of this method is that it can quickly and accurately identify simple and common malicious code. However, it performs relatively poorly in terms of accuracy and efficiency when dealing with complex and advanced malware.

Semantics-based methods typically employ program analysis techniques. Program analysis is a technique that abstracts program behavior into mathematical models for further presentation and analysis. Compared to syntax-based methods, this approach can better identify complex and advanced malware and extract useful information from it. Through program analysis, researchers can better extract semantic information during the execution of malware to understand its behavior and characteristics, which are often expressed through graphs and data flow methods.

### Feasible Methods

A feasible approach for this system involves first performing static analysis on Android applications to extract function call graphs and automatically annotate these graphs using an existing [sensitive API dataset](https://apichecker.github.io/). Then, design a graph convolutional neural network (GCN) model and use One-Hot encoding to implement feature encoding for the function call graph, initializing the feature vectors for each node in the GCN model. Finally, train the model using existing datasets. For static analysis of Android applications, tools such as [Androguard](https://github.com/androguard/androguard) can be used. Malicious Android applications can be downloaded from the [VirusShare](https://virusshare.com/) platform, while normal Android applications can be obtained from the [AndroZoo](https://androzoo.uni.lu/) platform.

![architecture](img/architecture-en.svg)

### Task Requirements

1. Review relevant literature to understand the principles of malware detection.
2. Design and implement a method for extracting function call graphs from Android applications and automatically annotating sensitive API calls.
3. Design and implement a method for training an Android malware detection model based on a graph convolutional neural network (GCN).
4. Design an experimental analysis method, requiring no fewer than 50 test samples for both malicious and normal applications, to evaluate the model's accuracy, recall rate, and F1 score.
5. Based on group division of labor, write individual course reports to form a comprehensive project report.
