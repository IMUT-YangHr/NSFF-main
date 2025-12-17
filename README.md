# NSFF: Noise and Semantic Features Fusion for AI-Generated Image Detection
This paper is currently under revision. We promise that, regardless of whether this paper is accepted, this codebase will remain open source permanently. Once this paper is accepted, we will publish the link here.
## Author
Haoran Yang, Ruiqiang Ma, Gang Wang, Jien Kato
## Method
![](https://github.com/IMUT-YangHr/NSFF-main/blob/main/NSFF.png "NSFF")

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
