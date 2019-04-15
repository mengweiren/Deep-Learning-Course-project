# Deep-Learning-Course-project
Deep Learning Course project based on VON ([Project Page](http://von.csail.mit.edu) |  [Paper](http://arxiv.org/abs/1812.02725)
)

Visual Object Networks: Image Generation with Disentangled 3D Representation.<br/>
[Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/),
 [Zhoutong Zhang](https://www.csail.mit.edu/person/zhoutong-zhang), [Chengkai Zhang](https://scholar.google.com/citations?user=rChGGwgAAAAJ&hl=en), [Jiajun Wu](https://jiajunwu.com/), [Antonio Torralba](http://web.mit.edu/torralba/www/), [    Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html), [William T. Freeman](http://billf.mit.edu/).<br/>
MIT CSAIL and Google Research.<br/>
In NeurIPS 2018.

## Prerequisites
- Linux (tested on Ubuntu 16.04/ 18.04)
- Python3.6
- Anaconda3
- NVCC & GCC (tested with gcc 5.4.0)
- PyTorch 0.4.1 (does not support 0.4.0)
- Currently (tested with Nvidia RTX GPU series)

### Installation
- Install PyTorch 0.4.1+ and torchvision from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by the following:
```bash
conda create --name von --file pkg_specs.txt
source activate von
```

- Compile the rendering kernel by running the following:
```bash
bash install.sh
```

### Model Training
- Download the training dataset (distance functions and images)
```bash
wget http://von.csail.mit.edu/data/data.tar
```

- To train a 3D generator:
```bash
python train.py --gpu_ids ${GPU_IDS} \
                  --display_id 1000 \
                  --dataset_mode df \
                  --model 'shape_gan' \
                  --class_3d ${CLASS} \
                  --checkpoints_dir ${CHECKPOINTS_DIR} \
                  --niter 250 --niter_decay 250 \
                  --batch_size 8 \
                  --save_epoch_freq 10 \
                  --suffix {class_3d}_{model}_{dataset_mode}
```
Specify the GPU_ID, CLASS (car or chair), checkpoints_dir. 

- To train a 2D texture network using ShapeNet real shapes:
```bash
python train.py --gpu_ids ${GPU_IDS} \
  --display_id 1000 \
  --dataset_mode 'image_and_'${DATASET} \
  --model 'texture_real' \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --class_3d ${CLASS} \
  --random_shift --color_jitter \
```
Specify the GPU_ID, CLASS (car or chair), checkpoints_dir. 

- To test the model by generating shapes and images:
```bash
python test.py --gpu_ids ${GPU_IDS} \
  --results_dir ${RESULTS_DIR} \
  --model2D_dir ${MODEL2D_DIR} \
  --model3D_dir ${MODEL3D_DIR} \
  --class_3d ${CLASS} \
  --phase 'val' \
  --dataset_mode 'image_and_df \
  --model 'test'  \
  --n_shapes ${NUM_SHAPES} \
  --n_views ${NUM_SAMPLES} \
  --reset_texture \
  --reset_shape \
  --suffix ${CLASS}_${DATASET}\
  --render_25d --render_3d
```  
Specify the GPU_ID, RESULTS_DIR, MODEL2D_DIR, MODEL3D_DIR, CLASS, NUM_SHAPES, NUM_SAMPLES
If you are using our pretrained models, please specify MODEL2D_DIR as './checkpoints/0411models/models_2D/car_df/latest', and MODEL3D_DIR as './checkpoints/0411models/models_3D/car_df'. (Currently only Car is supported, and there are some results under /results folder)


### Citation

If you find this useful for your research, please cite the following paper.
```
@inproceedings{VON,
  title={Visual Object Networks: Image Generation with Disentangled 3{D} Representations},
  author={Jun-Yan Zhu and Zhoutong Zhang and Chengkai Zhang and Jiajun Wu and Antonio Torralba and Joshua B. Tenenbaum and William T. Freeman},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

```
### Acknowledgements
This repository is for educational use only. The code borrows from [VON](https://github.com/junyanz/VON.git). 



