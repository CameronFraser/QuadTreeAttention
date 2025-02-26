# Updated Readme

Fork: This fork focuses on feature matching and updates the packaging system to not use sys path hacks

Run `make local` to build the docker image and `make bash` to run it

Once inside the container run `python match_pair.py -h` to get the help output:
```
usage: match_pair.py [-h] [--weight WEIGHT] [--config_path CONFIG_PATH] [--input_1st INPUT_1ST]
                     [--input_2nd INPUT_2ND] [--output_file OUTPUT_FILE]
                     [--ransac_reproj_thresh RANSAC_REPROJ_THRESH] [--ransac_confidence RANSAC_CONFIDENCE]
                     [--ransac_max_iters RANSAC_MAX_ITERS]

QuadTreeAttention demo

optional arguments:
  -h, --help            show this help message and exit
  --weight WEIGHT       Path to the checkpoint. (default: /workspace/weights/indoor.ckpt)
  --config_path CONFIG_PATH
                        Path to the config. (default:
                        /workspace/FeatureMatching/loftr/configs/loftr/indoor/loftr_ds_quadtree.py)
  --input_1st INPUT_1ST
                        1st image. (default: None)
  --input_2nd INPUT_2ND
                        2nd image. (default: None)
  --output_folder OUTPUT_FOLDER
                        Folder to save .mat and plot in. (default: /workspace/data/output)
  --ransac_reproj_thresh RANSAC_REPROJ_THRESH
                        Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar
                        line in pixels, beyond which the point is considered an outlier and is not used for
                        computing the final fundamental matrix. (default: 0.5)
  --ransac_confidence RANSAC_CONFIDENCE
                        Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level
                        of confidence (probability) that the estimated matrix is correct. In the range 0..1
                        exclusive. (default: 0.999)
  --ransac_max_iters RANSAC_MAX_ITERS
  ```

  To run a demo run `python match_pair.py --input_1st ./data/images/pair_1.jpg --input_2nd ./data/images/pair_2.jpg --output_folder ./data/output`

# Original Readme

This repository contains codes for quadtree attention. This repo contains codes for feature matching, image classficiation, object detection and semantic segmentation.

<div align="center">
  <img width="800", src="./teaser.png">
</div>


# Installation
1. Compile the quadtree attention operation
```cd QuadTreeAttention&&python setup.py install```
2. Install the package for each task according to each README.md in the separate directory. 

# Model Zoo and Baselines
We provide baselines results and model zoo in the following.

### Feature matching
#### News! QuadTree Attention achieves the best single model performance among all public available pretrained models in image matching chanllenge 2022. Please refer to this [[post]](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328805).

- Quadtree on Feature matching

| Method           | AUC@5 | AUC@10 | AUC@20 | Model |
|------------------|:----:|:-----:|:------:|:-------:|
| ScanNet     | 24.9  |  44.7 |  61.8 |[[Google]](https://drive.google.com/file/d/1pSK_8GP1WkqKL5m7J4aHvhFixdLP6Yfa/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/indoor.ckpt) |
| Megadepth   | 53.5  |  70.2 |  82.2 |[[Google]](https://drive.google.com/file/d/1UOYdzbrXHU9kvVy9tscCCO7BB3G4rWK4/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt) |

### Image classification

- Quadtree on ImageNet-1K

| Method           | Flops | Acc@1 | Model |
|------------------|:----:|:-----:|:-----:|
| Quadtree-B-b0        |  0.6 |  72.0 |  [[Google]](https://drive.google.com/file/d/13hBEBXXmTc3NI0WOqNE89Yd5GZf7wCJN/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_cls/b0.pth) |
| Quadtree-B-b1        |  2.3 |  80.0 |  [[Google]](https://drive.google.com/file/d/1NB1Yu0R7QQPmo2pgQGxDbcElRx2Nc5xj/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_cls/b2.pth) |
| Quadtree-B-b2        |  4.5 |  82.7 |  [[Google]](https://drive.google.com/file/d/1MTexxhDpRE9idpxwswZOGsqAlt9q2L2h/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_cls/b2.pth) |
| Quadtree-B-b3        | 7.8  |  83.8 |  [[Google]](https://drive.google.com/file/d/1Rx_JhGDKXKfOakY8n5HQgxuAyrjBA7jA/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_cls/b3.pth) |
| Quadtree-B-b4        | 11.5 |  84.0 |  [[Google]](https://drive.google.com/file/d/1AiPWGJYZdqz09PZER3JpZuMJYBR___MG/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_cls/b4.pth) |

### Object detection and instance segmentation

- Quadtree on COCO

#### Baseline Detectors


|   Method   | Backbone | Pretrain    | Lr schd | Aug | Box AP | Mask AP | Model    |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|:-------:|
|  RetinaNet | Quadtree-B-b0 | ImageNet-1K |    1x   |  No |  38.4  |    -    | [[Google]](https://drive.google.com/file/d/1EkzDVRqz6L_2ZzVkAj-byQIB4CCr3gNp/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_detection/b0.pth) |
|  RetinaNet | Quadtree-B-b1 | ImageNet-1K |    1x   |  No |  42.6  |    -    | [[Google]](https://drive.google.com/file/d/1xqZoptlHj1nWEVvUBqSOiIlEiy1Y5gws/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_detection/b1.pth) |
|  RetinaNet | Quadtree-B-b2 | ImageNet-1K |    1x   |  No |  46.2  |    -    | [[Google]](https://drive.google.com/file/d/1n6Zyvgdf4slhKMsG4CpckGDCNbB_48oi/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_detection/b2.pth) |
|  RetinaNet | Quadtree-B-b3 | ImageNet-1K |    1x   |  No |  47.3  |    -    | [[Google]](https://drive.google.com/file/d/1SVIWM9JFfW9a1jIjB8gdkpOaccSgKUGc/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_detection/b3.pth) |
|  RetinaNet | Quadtree-B-b4 | ImageNet-1K |    1x   |  No |  47.9  |    -    | [[Google]](https://drive.google.com/file/d/1nMUhg2N59FWqbIAbZcLwSe07L1OjPHiY/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_detection/b4.pth) |
| Mask R-CNN | Quadtree-B-b0 | ImageNet-1K |    1x   |  No |  38.8  |   36.5  | [[Google]](https://drive.google.com/file/d/1f_NGHygQC8Y-EVLeHE5T7UjZU5n6WJJs/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_maskrcnn/b0_1x.pth) |
| Mask R-CNN | Quadtree-B-b1 | ImageNet-1K |    1x   |  No |  43.5  |   40.1  | [[Google]](https://drive.google.com/file/d/1916HZGGzdfHVL2osTUHOlKKH-88NkbhE/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_maskrcnn/b1_1x.pth) |
| Mask R-CNN | Quadtree-B-b2 | ImageNet-1K |    1x   |  No |  46.7  |   42.4  | [[Google]](https://drive.google.com/file/d/1KhKVbslAUw6tbSHDxb6vZXLGsY__gcQ_/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_maskrcnn/b2_1x.pth) |
| Mask R-CNN | Quadtree-B-b3 | ImageNet-1K |    1x   |  No |  48.3  |   43.3  | [[Google]](https://drive.google.com/file/d/1_STW0pE1Gt-JrLdd-G1-skheb7XTHD3y/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_maskrcnn/b3_1x.pth) |
| Mask R-CNN | Quadtree-B-b4 | ImageNet-1K |    1x   |  No |  48.6  |   43.6 | [[Google]](https://drive.google.com/file/d/1jebBbBhtCHw3rmHM32Kf4Z8sDlCpaGr4/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_maskrcnn/b4_1x.pth) |

### Semantic Segmentation

- Quadtree on ADE20K

| Method       | Backbone   | Pretrain    | Iters | mIoU | Model |
|--------------|------------|-------------|-------|------|------|
| Semantic FPN | Quadtree-b0   | ImageNet-1K | 160K   | 39.9 |[[Google]](https://drive.google.com/file/d/1qTman3_vAnEJs8g_5CeOr8wtcCK2gvJM/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_seg/b0.pth) |
| Semantic FPN | Quadtree-b1  | ImageNet-1K | 160K   | 44.7 |[[Google]](https://drive.google.com/file/d/1SQKe9FmpmR__Fq0bNYEvO0T_5B--zs7b/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_seg/b1.pth) |
| Semantic FPN | Quadtree-b2 | ImageNet-1K | 160K   | 48.7 |[[Google]](https://drive.google.com/file/d/1pyyJWvXPRxApNiCRaR4c73nVQf-yCKQ2/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_seg/b2.pth) |
| Semantic FPN | Quadtree-b3  | ImageNet-1K | 160K   | 50.0 | [[Google]](https://drive.google.com/file/d/1odZkr2c0Oa8jJxr3TUsgKrHaLwuMInfp/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_seg/b3.pth) |
| Semantic FPN | Quadtree-b4  | ImageNet-1K | 160K   | 50.6 | [[Google]](https://drive.google.com/file/d/16ZBvzR51XUk3cpnJ6D1BgIvFS9oSzzh8/view?usp=sharing)/[[GitHub]](https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_seg/b4.pth) |

## Citation

```
@article{tang2022quadtree,
  title={QuadTree Attention for Vision Transformers},
  author={Tang, Shitao and Zhang, Jiahui and Zhu, Siyu and Tan, Ping},
  journal={ICLR},
  year={2022}
}
```
## License

The MIT License (MIT)

Copyright (c) 2022 Shitao Tang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
