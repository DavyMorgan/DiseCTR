# DiseCTR: Disentangled Interest Network for Out-of-Distribution CTR Prediction

This is our TensorFlow implementation for DiseCTR.

The code is tested under a Linux desktop with TensorFlow 2.3 and Python 3.8.10.




## Model Training

Use the following command to train a DiseCTR model on `Amazon` dataset: 

```
python examples/run_video_debias.py --dataset amazon
```

or on `Wechat` dataset: 

```
python examples/run_video_debias.py --dataset wechat
```

or on `Kuaishou` dataset:

```
python examples/run_video_debias.py --dataset kuaishou
``` 

## Note

The implemention is based on *[DeepCTR](https://github.com/shenweichen/DeepCTR)*.