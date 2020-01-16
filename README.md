# ENMF

This is our implementation of Efficient Neural Matrix Factorization, which is a basic model of the paper:



*Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Yiqun Liu and Shaoping Ma. 2019. [An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.](http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf) 
In SIGIR'19.*


This is also the codes of the TOIS paper:

*Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020. [Efficient Neural Matrix Factorization without Sampling for Recommendation.](https://chenchongthu.github.io/files/TOIS_ENMF.pdf) 
In TOIS Vol. 38, No. 2, Article 14.*

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
numpages = {28}}
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


Last Update Date: May 9, 2019
