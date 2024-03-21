 # <p align="center"> <font color=#008000>TANGO</font>: <font color=#008000>T</font>ext-driven Photore<font color=#008000>a</font>listic a<font color=#008000>n</font>d Robust 3D Stylization via Li<font color=#008000>g</font>hting Dec<font color=#008000>o</font>mposition </p>

 #####  <p align="center"> [Yongwei Chen](https://cyw-3d.github.io/), Rui Chen, [Jiabao Lei](https://jblei.site/), [Yabin Zhang](https://ybzh.github.io/), [Kui Jia](http://kuijia.site/)</p>

#### <p align="center"> NeurIPS 2022 <font color=#dd0000>*Spotlight*</font></p>

#### <p align="center">[Paper](http://arxiv.org/abs/2210.11277) | [Project Page](http://cyw-3d.github.io/tango) </p>

<p align="center">
  <img width="100%" src="https://github.com/Gorilla-Lab-SCUT/tango/blob/main/tango_assets/stylization.gif"/>
</p>

### News
:trophy: **[Nov 11th 2022]** TANGO was selected  in NeurIPS *[spotlight papers](https://nips.cc/Conferences/2022/Schedule?type=Spotlight)*.

### Installation

**Note:** You can directly pull the image we uploaded to AliCloud
```
docker pull registry.cn-shenzhen.aliyuncs.com/baopin/t2m:1.7
```

### System Requirements
- Python >=3.7 and <=3.9
- CUDA 11
- Nvidia GPU with 12 GB ram at least
- Open3d >=0.14.1
- the package of clip (https://github.com/openai/CLIP)

### Train
Call the below shell scripts to generate example styles. 
```bash
# shoe made of gold
./demo/run_shoe_gold.sh
# vase made of wicker 
./demo/run_vase_wicker.sh
# car made of wood
./demo/run_car_wood_origin.sh
# ...
```
The outputs will be saved to `results/demo`

### Validate
Call the below shell scripts to generate gif. 
```bash
# shoe made of gold
./demo/test_shoe_gold.sh
# vase made of wicker 
./demo/test_vase_wicker.sh
# car made of wood
./demo/test_car_wood_origin.sh
# ...
```
<!-- <p align="center">
  <img width="100%" src="./tango_assets/method.jpg"/>
</p> -->

## Citation
```
@article{chen2022tango,
  title={Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition},
  author={Chen, Yongwei and Chen, Rui and Lei, Jiabao and Zhang, Yabin and Jia, Kui},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={30923--30936},
  year={2022}
}
```
