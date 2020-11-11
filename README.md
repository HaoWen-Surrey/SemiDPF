# Semi-supervised Differentiable Particle Filters

This repository provides the source code for semi-supervised differentiable particle filters described in the paper "End-to-end semi-supervised learning for differentiable particle filters".

## Prerequisites

### Python packages

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

### Datasets
Download the dataset for Maze environments to the ./data/ folder from the link: https://tubcloud.tu-berlin.de/s/0fU32cq0ppqdGXe/download.

## Project & Script Descriptions

In the main repository folder, the following command needs to append the parent directory to the PYTHONPATH.

```
export PYTHONPATH="${PYTHONPATH}:../"
```

Then you can train and test the semi-supervised differentiable particle filters in Maze environments by running the following commands:

```
cd experiments; python main.py
```

### Scripts

Here are the descriptions for the scipts. 
- ```./experiments/main.py``` the main file for the training and test of SemiDPF.
- ```./methods/dpf.py``` the implementations of SemiDPF.
- ```./methods/rnn.py``` the implementations of LSTM baseline algorithm.
- ```./utils/data_utils.py``` the utility function to preprocess the input data.
- ```./utils/exp_utils.py``` the utility function for experiment setup.
- ```./utils/method_utils.py``` some auxiliary functions for implementation of SemiDPF.
- ```./utils/plotting_utils.py``` the utility function for plotting the experiment results.


## References
* A large amount of the SemiDPF code and folder structure was based on the implementation of Rico Jonschkowski [tu-rbo/differentiable-particle-filters](https://github.com/tu-rbo/differentiable-particle-filters)
* [Particle Filter Networks](https://github.com/AdaCompNUS/pfnet)
* [Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors](https://arxiv.org/abs/1805.11122)
* [Particle filter networks with application to visual localization](https://arxiv.org/abs/1805.08975)
