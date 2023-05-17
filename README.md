# DiseCTR: Disentangled Interest Network for Out-of-Distribution CTR Prediction

This is our TensorFlow implementation for DiseCTR.

The code is tested under a Linux desktop with TensorFlow 2.3 and Python 3.8.10.




## Model Training
Unzip the data files in `data` folder.

Change to the script directory.
```
cd examples
```

Use the following command to train a DiseCTR model on `Amazon` dataset: 

```
python run_disectr.py --dataset amazon --name amazon_test
```

or on `Wechat` dataset: 

```
python run_disectr.py --dataset wechat --name wechat_test
```

or on `Kuaishou` dataset:

```
python run_disectr.py --dataset kuaishou --name kuaishou_test
``` 

## Note

The implemention is based on *[DeepCTR](https://github.com/shenweichen/DeepCTR)*.