# Full-face
The Pytorch Implementation of "It’s written all over your face: Full-face appearance-based gaze estimation".

This is the implementated version in our survey "Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark".
Please refer our paper or visit our benchmark website <a href="http://phi-ai.org/project/Gazehub/" target="_blank">*GazeHub*</a> for more information.
The performance of this version is reported in them.

To know more detail about the method, please refer the origin paper.

We recommend you to use the data processing code provided in <a href="http://phi-ai.org/project/Gazehub/" target="_blank">GazeHub</a>.
You can use the processed dataset and this code for directly running.

## Introduction
We provide two similar projects for leave-one-person-out evaluation and common training-test split.
They have the same architecture but different started modes.

Each project contains following files/folders.
- `model.py`, the model code.
- `train.py`, the entry for training.
- `test.py`, the entry for testing.
- `config/`, this folder contains the config of the experiment in each dataset. To run our code, **you should write your own** `config.yaml`. 
- `reader/`, the code for reading data. You can use the provided reader or write your own reader.

## Getting Started
### Writing your own *config.yaml*.

Normally, for training, you should change 
1. `train.save.save_path`, The model is saved in the `$save_path$/checkpoint/`.
2. `train.data.image`, This is the path of image.
3. `train.data.label`, This is the path of label.
4. `reader`, This indicates the used reader. It is the filename in `reader` folder, e.g., *reader/reader_mpii.py* ==> `reader: reader_mpii`.

For test, you should change 
1. `test.load.load_path`, it is usually the same as `train.save.save_path`. The test result is saved in `$load_path$/evaluation/`.
2. `test.data.image`, it is usually the same as `train.data.image`.
3. `test.data.label`, it is usually the same as `train.data.label`.
 
### Training.

In leaveone folder, you can run
```
python train.py config/config_mpii.yaml 0
```
This means the code running with `config_mpii.yaml` and use the `0th` person as test set.

You also can run
```
bash run.sh train.py config/config_mpii.yaml
```
This means the code will perform leave-one-person-out training automatically.   
`run.sh` performs iteration, you can change the iteration times in `run.sh` for different datasets, e.g., set the iteration times as `4` for four-fold validation.

In the traintest folder, you can run
```
python train.py config/config_mpii.yaml
```

### Testing.
In leaveone folder, you can run
```
python test.py config/config_mpii.yaml 0
```
or
```
bash run.sh train.py config/config_mpii.yaml
```

In the traintest folder, you can run
```
python test.py config/config_mpii.yaml
```

### Result
After training or test, you can find the result from the `save_path` in `config_mpii.yaml`. 


## Citation
```
@inproceedings{Zhang_2017_CVPRW,
	title={It’s written all over your face: Full-face appearance-based gaze estimation},
	author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
	booktitle={The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
	pages={2299--2308},
	month={July},
	year={2017},
	organization={IEEE}
}

@inproceedings{Cheng2021Survey,
    title={Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark},
    author={Yihua Cheng, Haofei Wang, Yiwei Bao, Feng Lu},
    booktitle={arxiv}
    year={2021}
}
```
## Contact 
Please email any questions or comments to yihua_c@buaa.edu.cn.
