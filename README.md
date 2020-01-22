# Pixel-level-Pavement-Defects-Segmentation-Based-on-Various-Loss-Functions
Improved Pixel-level Pavement Defects Segmentation Based on Various Loss Functions
  
In this place we store U-Net based neural network implementation used for investigation. It containes model and loss function implementation, using Keras with Tensorflow backend. Another architectures might be derived using this codebase. Few ideas: autoencoders with triple convolutional, quadruple convolutional operation, residual connections.

Investigation is made on CrackForest dataset [1], [2]. Data examples (label is invested for visualization reasons):  
![alt-text-1](https://github.com/rytisss/Pixel-level-Pavement-Defects-Segmentation-Based-on-Various-Loss-Functions/blob/master/res/035_image.png "image") ![alt-text-2](https://github.com/rytisss/Pixel-level-Pavement-Defects-Segmentation-Based-on-Various-Loss-Functions/blob/master/res/035_label.png "label")

Architecture used for research (script for PlotNeuralNet project used to visualize network [3]):  
![alt text](https://github.com/rytisss/Pixel-level-Pavement-Defects-Segmentation-Based-on-Various-Loss-Functions/blob/master/res/4LayerUnet.png)
  
Link to the video results models trained with different loss functions on all randomly formed datasets: https://www.youtube.com/playlist?list=PL5dj7GxMk-6wWaBCS0EigijdeDQVvBz9m
  
Link to randomly formed (from CrackForest [1], [2]) datasets used for research: https://drive.google.com/file/d/1b6MjIp85aHUIp5ZOO-v-v5cRv2xMFwAO/view?usp=sharing
  
References:  
[1] Shi, Y.; Cui, L.; Qi, Z.; Meng, F.; Chen, Z. Automatic road crack detection using random structured forests. IEEE Trans. Intell. Transp. Syst. 2016, 17, 3434–3445.  
[2] Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Y. Pavement Distress Detection Using Random Decision Forests; Springer, 2015;  
[3] Iqbal, H. HarisIqbal88/PlotNeuralNet v1.0.0 (Version v1.0.0). 2018. (url: https://github.com/HarisIqbal88/PlotNeuralNet)


