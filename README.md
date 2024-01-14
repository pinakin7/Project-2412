# Privacy-Preserving Generative Models: A Comparative Analysis

This repository contains the research paper "Privacy-Preserving Generative Models: A Comparative Analysis" by Aneri Gandhi and Preet Viradiya from the University of Toronto. The paper investigates the training of image generative models with a focus on differential privacy, specifically the Conditional Generative Adversarial Network (CGAN), Conditional Variational Autoencoder (CVAE), and the Diffusion Model. The study aims to shed light on the delicate balance between privacy and accuracy, providing insights into how increasing the privacy budget affects the quality of generated images.

## Abstract

The research utilizes the MNIST dataset to numerically measure the privacy-accuracy trade-off using parameters from ϵ − δ differential privacy and the Fréchet Inception Distance (FID) score. The experiments provide insights into the impact of differential privacy on the quality of generated images, offering valuable implications for the deployment of privacy-preserving generative models.

## Methodology

The experiments adopt high-performing architectures for the three models: CGAN, CVAE, and Diffusion, trained with differential privacy parameters set to ϵ = 5, 10, 40, and δ = 1e − 5 using the DP-SGD optimizer. The training data is drawn from the MNIST dataset, comprising 60,000 black and white images distributed across digit classes 0 to 9. The models' performance is evaluated by calculating the FID score using 1000 generated images, with 100 images sampled from each class.

## Results

The results support previous research by demonstrating that non-differentially private generative models, specifically the Diffusion model, outperform their counterparts, VAE and GAN. However, the robustness analysis against membership inference attacks revealed unexpected accuracy results, raising concerns about potential TensorFlow API issues or overfitting in the shadow logistic regression model. The study also highlights the need for further investigation into optimal model selection for industrial applications, considering training speed and resource requirements.

## Conclusion

In conclusion, the research provides valuable insights into the trade-offs and challenges of using differentially private generative models. The study emphasizes the importance of balancing privacy and data quality, offering implications for the practical deployment of privacy-preserving generative models.
