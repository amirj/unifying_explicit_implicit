# Unifying Explicit and Implicit Feedback for Rating Prediction and Ranking Recommendation Tasks

This is the implementation of the following paper:

```
@InProceedings{unifying_exp0licit_implicit,
  author    = {Amir H. Jadidinejad and Craig Macdonald and Iadh Ounis},
  title     = {Unifying Explicit and Implicit Feedback for Rating Predictionand Ranking Recommendation Tasks},
  publisher = {ACM},
  booktitle = {International Conference on the Theory of Information Retrieval (ICTIR)},
  year      = {2019},
}
```

## Requirements
* [pytorch](https://github.com/pytorch/pytorch) (1.0.1)
* [spotlight](https://github.com/maciejkula/spotlight) (0.1.5)
* [pytrec-eval](https://github.com/cvangysel/pytrec_eval) (0.3)


## Results
The following table summarizes the results of the proposed model ($\Phi_{D_{i}^{*}}$) compared to the baseline explicit ($\Phi_{D_{e}}$) and implicit ($MF_{bpr}$) models:
![results](results.png)

See [the paper](#) for more details.

## How to reproduce?
Use the corresponding Jupyter notebook to reproduce the results of each dataset:
* [GoodReads Children](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_goodreads_children.ipynb)
* [GoodReads Comics & Graphics](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_goodreads_comics.ipynb)
* [GoodReads Fantasy & Paranormal](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_goodreads_Fantasy.ipynb)
* [Steam](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_steam.ipynb)
* [Douban](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_douban.ipynb)
* [Dianping](https://github.com/amirj/unifying_explicit_implicit/blob/master/weak_recsys_dianping.ipynb)

