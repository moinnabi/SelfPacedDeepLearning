# SelfPacedDeepLearning

Introduction:
============
Self-paced deep learning is a framework for weakly supervised object detection with deep ConvNets. 


Installation
============
1- Install Caffe and Fast R-CNN [1] code

The code is heavily depend on fast-rcnn code, so this code should be installed first. See the details from here: https://github.com/rbgirshick/fast-rcnn

2- Download the training, validation, test data and VOCdevkit, and create symlinks for the PASCAL VOC dataset.

3- Download pre-computed Selective Search object proposals
```bash
cd $SP_FRCN_ROOT
./data/scripts/fetch_selective_search_data.sh
```
4- Download pre-computed Weakly Supervised detectors
```bash
cd $SP_FRCN_ROOT
./data/scripts/fetch_SP_fast_rcnn_models.sh
```

Usage
=====

Test
----
```bash
./tools/test_net.py --gpu 0 --def $SP_FRCN_ROOT/SP_models/voc/test_voc.prototxt --net $SP_FRCN_ROOT/output/default/voc_2007_trainval/SP_ws_f_rcnn_voc_iter_33941.caffemodel
```

Train
-----
First, put the initialization models and prototexts from ```$SP_FRCN_ROOT/SP_models/voc``` into output directory, e.g.: ```$SP_FRCN_ROOT/output/default/voc_2007_trainval```

Then, run:
```bash
./tools/train_net.py --gpu 0 --solver $SP_FRCN_ROOT/output/default/voc_2007_trainval/solver_voc.prototxt --weights $SP_FRCN_ROOT/output/default/voc_2007_trainval/SP_ws_f_rcnn_voc_iter_0.caffemodel
```

How to Cite
===========
If you use this code or the pretrained models in your research,
please cite:

```
@article{sangineto2016self,
  title={Self Paced Deep Learning for Weakly Supervised Object Detection},
  author={Sangineto, Enver and Nabi, Moin and Culibrk, Dubravko and Sebe, Nicu},
  journal={arXiv preprint arXiv:1605.07651},
  year={2016}
}
```

Reference(s)
===========
[1] R. Girshick. Fast R-CNN. arXiv:1504.08083, 2015.


