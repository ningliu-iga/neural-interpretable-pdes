# Neural Interpretable PDEs (NIPS)

![NIPS architecture.](https://github.com/ningliu-iga/neural-interpretable-pdes/blob/main/NIPS_architecture.png)

This repository houses the code for our ICML 2025 paper:
- Neural Interpretable PDEs: Harmonizing Fourier Insights with Attention for Scalable and Interpretable Physics Discovery

**Abstract**: Attention mechanisms have emerged as transformative tools in core AI domains such as natural language processing and computer vision. Yet, their largely untapped potential for modeling intricate physical systems presents a compelling frontier. Learning such systems often entails discovering operators that map between functional spaces using limited instances of function pairs---a task commonly framed as a severely ill-posed inverse PDE problem. In this work, we introduce Neural Interpretable PDEs (NIPS), a novel neural operator architecture that builds upon and enhances Nonlocal Attention Operators (NAO) in both predictive accuracy and computational efficiency. NIPS employs a linear attention mechanism to enable scalable learning and integrates a learnable kernel network that acts as a channel-independent convolution in Fourier space. As a consequence, NIPS eliminates the need to explicitly compute and store large pairwise interactions, effectively amortizing the cost of handling spatial interactions into the Fourier transform. Empirical evaluations demonstrate that NIPS consistently surpasses NAO and other baselines across diverse benchmarks, heralding a substantial leap in scalable, interpretable, and efficient physics learning.

## Requirements
- [PyTorch](https://pytorch.org/)


## Running experiments
To run the 2D Darcy example in the NIPS paper
```
python3 NIPS_Darcy.py
python3 LinearNAO_Darcy.py
```
To run the Mechanical MNIST example in the NIPS paper
```
python3 NIPS_MMNIST.py
```
## Datasets
We provide the Darcy and MMNIST datasets that are used in the paper.

- [Darcy and MMNIST datasets](https://drive.google.com/drive/folders/1-HA5uPMBHEH96sRcdzKaF7dyn8KQv8kG?usp=sharing)

## Citation
Neural Interpretable PDEs are an enhanced version of [Nonlocal Attention Operator (NAO)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce5b4f79f4752b7f8e983a80ebcd9c7a-Abstract-Conference.html), with learnable Fourier kernel and linear attention. If you find our models useful, please consider citing our papers:
```
@inproceedings{liu2025nips,
  title={Neural Interpretable PDEs: Harmonizing Fourier Insights with Attention for Scalable and Interpretable Physics Discovery},
  author={Liu, Ning and Yu, Yue},
  booktitle={Proceedings of the 42th International Conference on Machine Learning (ICML 2025)}
}
@inproceedings{yu2024nonlocal,
  title={Nonlocal Attention Operator: Materializing Hidden Knowledge Towards Interpretable Physics Discovery},
  author={Yu, Yue and Liu, Ning and Lu, Fei and Gao, Tian and Jafarzadeh, Siavash and Silling, Stewart},
  booktitle={Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)}
}
```
