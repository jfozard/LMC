# LMC

## Method

This entry consisted of a UNet (Rommeberger 2015), with an efficientnet-b7 (Tan 2020) encoder and sigmoid output activation function, implemented using the segmentation_models_pytorch library [https://github.com/qubvel/segmentation_models.pytorch].
The model was initialized with weights from pre-training on Imagenet, and used to predict all four output channels (nucleus, mitochondria, actin and tubulin). For each training sample, a crop of size 512x512, a z-slice, and one output channel were all randomly selected. Inputs were normalized to lie in the range [0,1], and per-image whitening was applied. Model outputs were compared with percentile-normalized ground truth images using a loss of the form $L = \lambda_1 L_1 + \lambda_2 L_2 + \lambda_{PCC} L_{PCC}$, where $L_1$ and $L_2$ are the standard L1 and mean-square errors, and $L_{PCC}(x,y) = \frac{ \sum_{i,j} (x_{i,j}-\bar{x})(y_{i,j}-\bar{y})}{\sqrt{\sum_{i,j} (x_{i,j}-\bar{x})^2}\sqrt{\sum_{i,j} (y_{i,j}-\bar{y})^2} + 10^{-6}}$. Losses were weighted according to $\lambda_1 = 0.5, \lambda_2=1.0, \lambda_{PCC} = 1$.This loss was optimized using the Adam optimizer, with batch-size 4, learning rate $10^{-4}$, and default pytorch Adam parameters $( \beta_1 = 0.9, \beta_2 = 0.999 )$. 

One of the input studies (Study 11) appears to be at least partially mislabelled, with the input brightfield images closely resembling the nucleus channel, and was omitted from the training dataset. The remaining studies were split randomly (on a per-image basis) into 90% training data and 10% test data.

All model training was performed using either a single RTX3090 GPU, each with 24GB VRAM. Initial model optimization proceeded for 140 epochs, taking approximately 8.5 hours.

As there were many fewer studies with actin and tubulin output images, the trained model was further fine-tuned twice, once on the data subset containing actin output images, and once on the subset containing tubulin output images. This fine-tuning was for 200 epochs taking 3.5 hours (tubulin), and 199 epochs taking 10 minutes (actin).

As the model is convolutional, model predictions were performed by applying the network to the whole input image, with appropriate zero-padding of images to a multiple of 32. For the nucleus and mitochondria channels, 8-way test-time augmentation (horizonal flip and/or rotations of 0,90,180 and 270 degrees) with mean averaging were applied, using the ttach library (https://github.com/qubvel/ttach). For the actin and tubulin output channels, the fine-tuned models were used, with no test-time augmentation.

Checkpoints are available at https://huggingface.co/JFoz/LMC_baseline

### UNet Architecture

<img src="https://github.com/jfozard/LMC/assets/4390954/78f58534-7b6f-49be-a5fe-1f1e612857e4" width="75%" />

### MBConv Block


<img src="https://github.com/jfozard/LMC/assets/4390954/f2d632a0-c51b-42bd-8478-cd5ffac4cd32" width="30%" />

### Inference

<img src="https://github.com/jfozard/LMC/assets/4390954/fd9c902e-f118-46c1-940c-76f938a8ec2d" width="50%" />

## Sample input

![image_0_BF_z0 ome](https://github.com/jfozard/LMC/assets/4390954/d219bfdd-389c-4b43-b152-7486e699b034)

Sample nucleus GT

![image_0_nucleus_gt ome](https://github.com/jfozard/LMC/assets/4390954/65c0b129-9809-4243-bd73-6426b9ac4128)

Sample nucleus output

![image_0_nucleus ome](https://github.com/jfozard/LMC/assets/4390954/d504757d-6ed5-4905-9a2e-3110acb9162e)

Sample tubulin output

![image_0_tubulin ome](https://github.com/jfozard/LMC/assets/4390954/4d478b6d-9bb2-4f7e-be04-d6cc144127cd)
