# NSFF: Noise and Semantic Features Fusion for AI-Generated Image Detection
This paper is currently under revision. We promise that, regardless of whether this paper is accepted, this codebase will remain open source permanently. Once this paper is accepted, we will publish the link here.
This repository is a simplified demo of our project.
## Author
Haoran Yang, Ruiqiang Ma, Gang Wang, Jien Kato
## Method
![](https://github.com/IMUT-YangHr/NSFF-main/blob/main/NSFF.png "NSFF")
## Requirements
-   Python 3.8
-   Pytorch 2.0.1+cu118
-  CUDA 11.8
## Train and Test
To simplify the training and evaluation process, we provide a one-stop running procedure. You can complete the training and evaluation directly on your local machine by running `AIGID.py`, or by running it using the command below. 

`python AIGID.py`

We placed the weight checkpoints for the classification model in the `weight` directory.
## Dataset
**Training set**：Extract from [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file).
| Generator | 0_real | 1_fake |
| :------: | :-------: | :------: |
| ProGAN | 500 | 500 |
| ADM | 500 | 500 |

**Testing set**：Extract from [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file), [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), [FakeBench](https://github.com/Yixuanli423/FakeBench), [Chameleon](https://github.com/shilinyan99/AIDE?tab=readme-ov-file).

| Source | Generator | 0_real | 1_fake |
| :------: | :------: | :-------: | :------: |
|AIGCDetectBenchMark| BigGAN | 500 | 500 |
|| CycleGAN | 500 | 500 |
|| GauGAN | 500 | 500 |
|| StyleGAN | 500 | 500 |
|| StyleGAN2 | 500 | 500 |
|| VQDM | 500 | 500 |
|| Glide | 500 | 500 |
|| Midjourney | 500 | 500 |
|| SDXL | 500 | 500 |
|| Wukong | 500 | 500 |
|UniversalFakeDetect| LDM-200-cfg | 500 | 500 |
|FakeBench| DALLE-3 | 200 | 200 |
|| CogView2 | 200 | 200 |
|Chameleon| Chameleon | 500 | 500 |
## Copyright Notice
### Regarding DINOv2
We hereby declare that we have only modified line 58 of the source code in `dinov2/hub/backbone.py`, and the modified content is as follows:
`state_dict = torch.hub.load_state_dict_from_url(url=url,model_dir="/NSFF-main/dinov2/weight", map_location="cpu")`
### Patent Declaration
Our method is patented (**2025107441785**) and protected by copyright law. **This patent primarily targets software or computer devices developed based on our method, and its content is completely independent of DINOv2.**

This patent covers only one method for generating image detection. While this method is inspired by DINOv2, it does not include the DINOv2 algorithm, technology, or implementation details.
### License Notice
Our license (**GPL-3.0**) applies only to our own method and does not involve the DINOv2 repository.
You are free to:
- View, download, and use our code for personal study, research, or evaluation.
- Modify the source code and distribute your modified versions, but must retain this statement.

Without express written permission, you may not:
- Use our code for any commercial purpose, including but not limited to sale, rental, or provision as part of a commercial product.
- Provide technical support or services based on our code to third parties in any form for compensation.

This repository is provided "as is," and the authors assume no responsibility.

The above **Copyright Notice** has been incorporated into `AIGID.py` and `AIGIDetection_custom_preprocessing.py`.
## Acknowledgement
This repository is based on [DINOv2](https://github.com/facebookresearch/dinov2), and we have also referenced [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file), and [AIDE](https://github.com/shilinyan99/AIDE?tab=readme-ov-file). Thanks for their wonderful works.
## other
If you have any questions or suggestions, please leave a message.
