# ENMF

This is our implementation of Efficient Neural Matrix Factorization, which is a basic model of the paper:



*Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Yiqun Liu and Shaoping Ma. 2019. [An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.](http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf) 
In SIGIR'19.*


This is also the codes of the TOIS paper:

*Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020. [Efficient Neural Matrix Factorization without Sampling for Recommendation.](https://chenchongthu.github.io/files/TOIS_ENMF.pdf) 
In TOIS Vol. 38, No. 2, Article 14.*

The slides of this work has been uploaded and a chinese version video presentation can be found at [Demo](https://www.bilibili.com/video/BV1Z64y1u7GK?from=search&seid=10581986304255794319)

**Please cite our SIGIR'19 paper or TOIS paper if you use our codes. Thanks!**

```
@inproceedings{chen2019efficient,
  title={An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation},
  author={Chen, Chong and Zhang, Min and Wang, Chenyang and Ma, Weizhi and Li, Minming and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={225--234},
  year={2019},
  organization={ACM}
}
```
```
@article{10.1145/3373807, 
author = {Chen, Chong and Zhang, Min and Zhang, Yongfeng and Liu, Yiqun and Ma, Shaoping}, 
title = {Efficient Neural Matrix Factorization without Sampling for Recommendation}, 
year = {2020}, 
issue_date = {January 2020}, 
publisher = {Association for Computing Machinery}, 
volume = {38}, 
number = {2}, 
issn = {1046-8188}, 
url = {https://doi.org/10.1145/3373807}, 
doi = {10.1145/3373807}, 
journal = {ACM Trans. Inf. Syst.}, 
month = jan, 
articleno = {Article 14}, 
numpages = {28}
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python
- Tensorflow
- numpy
- pandas


## Example to run the codes		

Train and evaluate the model:

```
python ENMF.py
```

## Comparison with the most recent methods （updating）

1. LightGCN (SIGIR 2020) [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf).

To be consistent to LightGCN, we use the same evaluation metrics (i.e., `Recall@K` and `NDCG@K`), use the same data Yelp2018 released in LightGCN (https://github.com/kuandeng/LightGCN).

The parameters of our ENMF on Yelp2018 are as follows:
```
parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout keep_prob')
parser.add_argument('--negative_weight', type=float, default=0.05,
                        help='weight of non-observed data')
```
Dataset: Yelp2018

|    Model    | Recall@20 | NDCG@20 |
| :---------: | :-------: | :----------: |
|     NGCF    |  0.0579   |    0.0477    |  
|     Mult-VAE     |  0.0584   |    0.0450    | 
|    GRMF    |  0.0571   |    0.0462    | 
|   LightGCN |  0.0649   |    0.0530    |
|   ENMF |  0.0650   |    0.0515    |








## Suggestions for parameters

Two important parameters need to be tuned for different datasets, which are:

```
parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout keep_prob')
parser.add_argument('--negative_weight', type=float, default=0.1,
                        help='weight of non-observed data')
```
                        
Specifically, we suggest to tune "negative_weight" among \[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]. Generally, this parameter is related to the sparsity of dataset. If the dataset is more sparse, then a small value of negative_weight may lead to a better performance.


Generally, the performance of our ENMF is better than existing state-of-the-art recommendation models like NCF, CovNCF, CMN, and NGCF. You can also contact us if you can not tune the parameters properly.



