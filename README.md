# Image Denoising with Convolutional Neural Networks

### Introduction
This project aims to address the challenge of photo noise caused by poor lighting or improper camera settings. By leveraging Convolutional Neural Networks (CNNs), specifically the Convolutional Blind Denoising Network (CBDNet), we developed an application capable of reducing noise in images while preserving their quality.

### Project Overview
The application addresses the limitations of traditional filtering methods, which often result in blurry images when attempting to remove complex natural noise. Our solution uses a two-step CNN approach to estimate noise levels and denoise images without compromising on resolution.

### Data Processing
Our model was trained and tested on a curated dataset composed of image pairs from three distinct sources, processed to generate 1920 image patches. This approach allowed for effective training and evaluation of the model's performance.

### Architecture
We adopted the CBDNet architecture, incorporating two networks: one for noise estimation and the other for denoising. This choice was based on extensive research and comparison with a baseline denoising autoencoder model.

### Results
Our model demonstrated promising denoising capabilities, particularly in monochromatic areas. However, it faced challenges in accurately processing images with dense patterns, occasionally blurring details.
