# ENMF

This is our implementation of Efficient Neural Matrix Factorization, which is a basic model of the paper:



*Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Yiqun Liu and Shaoping Ma. 2019. [An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.](http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf) 
In SIGIR'19.*


This is also the codes of the TOIS paper:

*Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020. [Efficient Neural Matrix Factorization without Sampling for Recommendation.](https://chenchongthu.github.io/files/TOIS_ENMF.pdf) 
In TOIS Vol. 38, No. 2, Article 14.*

The slides of this work has been uploaded. A chinese version instruction can be found at [Blog](https://zhuanlan.zhihu.com/p/107761829), and the video presentation can be found at [Demo](https://www.bilibili.com/video/BV1Z64y1u7GK?from=search&seid=10581986304255794319).

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

## Comparison with the most recent methods （updating）

Do the "state-of-the-art" recommendation models **really perform well?** If you want to see more comparison between our ENMF and any "state-of-the-art" recommendation models, feel free to propose an issue.

### 1. LightGCN (SIGIR 2020) [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf).

To be consistent with LightGCN, we use the same evaluation metrics (i.e., `Recall@K` and `NDCG@K`), use the same data Yelp2018 released in LightGCN (https://github.com/kuandeng/LightGCN).

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

### 2. NBPO (SIGIR 2020) [Sampler Design for Implicit Feedback Data by Noisy-label Robust Learning](https://doi.org/10.1145/3397271.3401155). 
This paper designs an adaptive sampler based on noisy-label robust learning for implicit feedback data. To be consistent with NBPO, we use the same evaluation metrics (i.e., `F1@K`, `NDCG@K`), use the same data Amazon-14core released in NBPO (https://github.com/Wenhui-Yu/NBPO). For fair comparison, we also set the embedding size as 50, which is utilized in the NBPO work.

The parameters of our ENMF on Amazon-14core are as follows:
```
parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout keep_prob')
parser.add_argument('--negative_weight', type=float, default=0.2,
                        help='weight of non-observed data')
```
Dataset: Amazon-14core

|    Model    | F1@5 | F1@10 |F1@20| NDCG@5 | NDCG@10 |NDCG@20|
| :---------: | :-------: | :----------: | :---------: | :-------: | :----------: | :----------: |
|     BPR    | 0.0326| 0.0317| 0.0275|0.0444| 0.0551| 0.0680| 
|     NBPO     |  0.0401| 0.0357| 0.0313|0.0555| 0.0655| 0.0810|
|   ENMF |  0.0419   |    0.0388    |0.0314|0.0566|0.0698|0.0823|

### 3. LCFN (ICML 2020)[Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters](https://arxiv.org/pdf/2006.15516v1.pdf)
To be consistent with LCFN, we use the same evaluation metrics (i.e., `F1@K`, `NDCG@K`), use the same data Movlelens-1m released in LCFN (https://github.com/Wenhui-Yu/LCFN). For fair comparison, we also set the embedding size as 128, which is utilized in the LCFN work.

The parameters of our ENMF on Movielens-1m (ml-lcfn) are as follows:
```
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout keep_prob')
parser.add_argument('--negative_weight', type=float, default=0.5,
                        help='weight of non-observed data')
```                       

Dataset: Movielens-1m (ml-lcfn)

|    Model    | F1@5 | F1@10 |F1@20| NDCG@5 | NDCG@10 |NDCG@20|
| :---------: | :-------: | :----------: | :---------: | :-------: | :----------: | :----------: |
|     GCMC    | 0.1166| 0.1437| 0.1564|0.2411| 0.2361| 0.2496| 
|     NGCF     |  0.1153| 0.1425| 0.1582|0.2367| 0.2347| 0.2511|
|     SCF     |  0.1189| 0.1451| 0.1600|0.2419| 0.2398| 0.2560|
|     CGMC     |  0.1179| 0.1431| 0.1573|0.2408| 0.2372| 0.2514|
|     LCFN     |  0.1213| 0.1482| 0.1625|0.2427| 0.2429| 0.2603|
|   ENMF |  0.1239   |    0.1512    |0.1640|0.2457|0.2475|0.2656|





