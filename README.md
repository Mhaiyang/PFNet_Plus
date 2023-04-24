# SSI2023_PFNet_Plus

## Distraction-Aware Camouflaged Object Segmentation
[Haiyang Mei](https://mhaiyang.github.io/), [Xin Yang](https://xinyangdut.github.io/), Yunduo Zhou, Ge-Peng Ji, Xiaopeng Wei, [Deng-Ping Fan](http://dpfan.net/)

[[Paper]()] [[Project Page](https://mhaiyang.github.io/SSI2023-PFNet-Plus/index.html)]

### Abstract
In this work, our goal is to design an effective and efficient camouflaged object segmentation (COS) model. To this end, we develop a bio-inspired framework, termed pyramid positioning and focus network (PFNet+), which mimics the process of predation in nature. Specifically, our PFNet+ contains three key modules, i.e., a context enrichment (CEn) module, a pyramid positioning module (PPM), and a focus module (FM). The CEn aims at enhancing the representation ability of backbone features via integrating contextual information for providing more discriminative backbone features. The PPM is designed to mimic the detection process in predation for positioning the potential target objects from a global perspective in a pyramid manner and the FM is then used to perform the identification process in predation for progressively refining the initial prediction via focusing on the ambiguous regions. Notably, in the FM, we develop a novel distraction mining strategy for distraction discovery and removal, to benefit the performance of estimation. Extensive experiments demonstrate that our PFNet+ runs in real-time (56 FPS) and outperforms 20 cutting-edge models on three challenging datasets under four standard metrics. The generalization capability of our PFNet+ is further demonstrated by the experiments on the other vision task (i.e., polyp segmentation).

### Citation
If you use this code, please cite:

```
@article{Haiyang:PFNet_Plus:2023,
    author = {Mei, Haiyang and Yang, Xin and Zhou, Yunduo and Ji, Ge-Peng and Wei, Xiaopeng and Fan, Deng-Ping.},
    title = {Distraction-Aware Camouflaged Object Segmentation},
    journal = {SCIENTIA SINICA Informationis (SSI)},
    year = {2023}
}
```

### Requirements
* PyTorch == 1.0.0
* TorchVision == 0.2.1
* CUDA 10.0  cudnn 7.2

### Train
Download 'resnet50-19c8e357.pth' at [here](https://download.pytorch.org/models/resnet50-19c8e357.pth), then run `train_four.py`.

### Test
Download trained model 'PFNet_Plus.pth' at [here](https://mhaiyang.github.io/SSI2023-PFNet-Plus/index.html), then run `infer.py`.

### License
Please see `License.txt`

### Contact
E-Mail: haiyang.mei@outlook.com
