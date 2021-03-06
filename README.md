<a href="https://tensorlayer.readthedocs.io/">
    <div align="center">
        <img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
    </div>
</a>




<!--- [![PyPI Version](https://badge.fury.io/py/tensorlayer.svg)](https://badge.fury.io/py/tensorlayer) --->
<!--- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)) --->
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/zsdonghao/tensorlayer2/master.svg)
[![Supported TF Version](https://img.shields.io/badge/TensorFlow-2.0.0%2B-brightgreen.svg)](https://github.com/tensorflow/tensorflow/releases)
[![Documentation Status](https://readthedocs.org/projects/tensorlayer2/badge/)](https://tensorlayer2.readthedocs.io/)
<!---  ([![Downloads](http://pepy.tech/badge/tensorlayer)](http://pepy.tech/project/tensorlayer)) --->
<!---  ([![Build Status](https://travis-ci.org/tensorlayer/tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/tensorlayer)) --->
<!---  [![CircleCI](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master.svg?style=svg)](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master) --->

<!---  [![Documentation Status](https://readthedocs.org/projects/tensorlayercn/badge/)](https://tensorlayercn.readthedocs.io/) --->
<!---  [![PyUP Updates](https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg)](https://pyup.io/repos/github/tensorlayer/tensorlayer/) --->
<!---  [![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg)](https://hub.docker.com/r/tensorlayer/tensorlayer/) --->
<!---  [![Codacy Badge](https://api.codacy.com/project/badge/Grade/d6b118784e25435498e7310745adb848)](https://www.codacy.com/app/tensorlayer/tensorlayer) --->


<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>


🚀🚀🚀🚀 Welcome to TensorLayer 2.0 !

We are working hard on updating TensorLayer 2.0 to support TensorFlow 2.0. Latest progress can be found [here](https://github.com/tensorlayer/tensorlayer/issues/900) and latest [doc](https://tensorlayer2.readthedocs.io). Feel free to discuss and [contribute](CONTRIBUTING.md)! Note that TensorLayer 2.0 is still under construction and not everything in this repository is ready yet.

TensorLayer is a novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers. It provides a large collection of customizable neural layers / functions that are key to build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the [ACM Multimedia Society](https://twitter.com/ImperialDSI/status/923928895325442049).


# Features

As deep learning practitioners, we have been looking for a library that can address various development
 purposes. This library is easy to adopt by providing diverse examples, tutorials and pre-trained models.
Also, it allow users to easily fine-tune TensorFlow; while being suitable for production deployment. TensorLayer aims to satisfy all these purposes. It has three key features:

- ***Simplicity*** : TensorLayer lifts the low-level dataflow interface of TensorFlow to *high-level* layers / models. It is very easy to learn through the rich [example codes](https://github.com/tensorlayer/awesome-tensorlayer) contributed by a wide community.
- ***Flexibility*** : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that help *low-level tuning* and *deep customization*.
- ***Zero-cost Abstraction*** : TensorLayer can achieve the *full power* of TensorFlow. The following table shows the training speeds of [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) using TensorLayer and native TensorFlow on a Tesla V100-DGXS-32GB.

	|   Mode    |       Lib       |  Data Format  | Max CPU Memory (MB) | Avg CPU Memory (MB) | Runtime (sec) |
	| :-------: | :-------------: | :-----------: | :-----------------: | :-----------------: | :-----------: |
	| AutoGraph | TensorFlow 2.0  | channel last  |        3370         |        3346         |      49       |
	|           | Tensorlayer 2.0 | channel last  |        3358         |        3367         |      50       |
	|   Graph   |      Keras      | channel last  |        3776         |        3775         |      62       |
	|   Eager   | TensorFlow 2.0  | channel last  |        3293         |        3284         |      65       |
	|           | TensorLayer 2.0 | channel last  |        3296         |        3293         |      65       |
	|           |     PyTorch     | channel first |        2562         |        2555         |      43       |



TensorLayer stands at a unique spot in the library landscape. Other wrapper libraries like Keras and TFLearn also provide high-level abstractions. They, however, often
hide the underlying engine from users, which make them hard to customize
and fine-tune. On the contrary, TensorLayer APIs are generally lightweight, flexible and transparent.
Users often find it easy to start with the examples and tutorials, and then dive
into TensorFlow seamlessly. In addition, TensorLayer does not create library lock-in through native supports for importing components from Keras, TFSlim and TFLearn.

TensorLayer has a fast growing usage among top researchers and engineers, from universities like
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and
University of Technology of Compiegne (UTC), and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

# Tutorials and Real-World Applications

You can find a large collection of tutorials, examples and real-world applications using TensorLayer within [examples](examples/) or through the following space:

<a href="https://github.com/tensorlayer/awesome-tensorlayer/blob/master/readme.md" target="\_blank">
	<div align="center">
		<img src="img/awesome-mentioned.png" width="40%"/>
	</div>
</a>

# Documentation

TensorLayer has extensive documentation for both beginners and professionals. The documentation is available in
both English and Chinese. Please click the following icons to find the documents you need:

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer2.readthedocs.io/)
<!--- [![Chinese Documentation](https://img.shields.io/badge/documentation-%E4%B8%AD%E6%96%87-blue.svg)](https://tensorlayercn.readthedocs.io/) --->
[![Chinese Book](https://img.shields.io/badge/book-%E4%B8%AD%E6%96%87-blue.svg)](http://www.broadview.com.cn/book/5059/)

If you want to try the experimental features on the the master branch, you can find the latest document
[here](https://tensorlayer2.readthedocs.io/en/latest/).

# Install



For latest code for TensorLayer 2.0, please build from the source. TensorLayer 2.0 has pre-requisites including TensorFlow 2, numpy, and others. For GPU support, CUDA and cuDNN are required.

```bash
# First clone the repository and change the current directory to the newly cloned repository
git clone https://github.com/zsdonghao/tensorlayer2.git
cd tensorlayer2

# Install virtualenv if necessary
pip install virtualenv
# Then create a virtualenv called `venv`
virtualenv venv

# Activate the virtualenv

## Linux:
source venv/bin/activate

## Windows:
venv\Scripts\activate.bat

# basic installation
pip install .

# ============= IF TENSORFLOW IS NOT ALREADY INSTALLED ============= #

# for a machine **without** an NVIDIA GPU
pip install -e ".[all_cpu_dev]"

# for a machine **with** an NVIDIA GPU
pip install -e ".[all_gpu_dev]"

```

If you want install TensorLayer 1.X, the simplest way to install TensorLayer 1.X is to use the **Py**thon **P**ackage **I**ndex (PyPI):

```bash
# for last stable version of TensorLayer 1.X
pip install --upgrade tensorlayer

# for latest release candidate of TensorLayer 1.X
pip install --upgrade --pre tensorlayer

# if you want to install the additional dependencies, you can also run
pip install --upgrade tensorlayer[all]              # all additional dependencies
pip install --upgrade tensorlayer[extra]            # only the `extra` dependencies
pip install --upgrade tensorlayer[contrib_loggers]  # only the `contrib_loggers` dependencies
```
<!---
Alternatively, you can install the latest or development version by directly pulling from github:

```bash
pip install https://github.com/tensorlayer/tensorlayer/archive/master.zip
# or
# pip install https://github.com/tensorlayer/tensorlayer/archive/<branch-name>.zip
```
--->

<!---
## Using Docker

The [TensorLayer containers](https://hub.docker.com/r/tensorlayer/tensorlayer/) are built on top of the official [TensorFlow containers](https://hub.docker.com/r/tensorflow/tensorflow/):

### Containers with CPU support

```bash
# for CPU version and Python 2
docker pull tensorlayer/tensorlayer:latest
docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest

# for CPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-py3
docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-py3
```

### Containers with GPU support

NVIDIA-Docker is required for these containers to work: [Project Link](https://github.com/NVIDIA/nvidia-docker)

```bash
# for GPU version and Python 2
docker pull tensorlayer/tensorlayer:latest-gpu
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

# for GPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-gpu-py3
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
```
--->

# Contribute

Please read the [Contributor Guideline](CONTRIBUTING.md) before submitting your PRs.

# Cite
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}
```

# License

TensorLayer is released under the Apache 2.0 license.
