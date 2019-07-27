# anomaly-aoutoencoding
Using autoencoders implemented with Keras for anomaly detection tasks

## Results
Two different autoencoder architectures were used, one with hidden layer, and another without hidden layer.
The reconstruction error of both models were used as prediction. These are the results:
![Alt text](pics/AUC_autoenc.png?raw=true "AUC comparision with CV")

It can be observed that there is little or no difference between both architectures.

## Architectures visualization

### Autoencoder without hidden layer
![Alt text](pics/smp_autoencoder.png?raw=true "")

### Autoencoder with hidden layer
![Alt text](pics/autoencoder.png?raw=true "")

## References
http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/  
https://arxiv.org/abs/1312.6114  
https://blog.keras.io/building-autoencoders-in-keras.html  
https://www.dataversity.net/fraud-detection-using-a-neural-autoencoder/  
https://deeplearning4j.org/tutorials/05-basic-autoencoder-anomaly-detection-using-reconstruction-error  
https://arxiv.org/abs/1811.05269  
https://www.kdd.org/kdd2017/papers/view/anomaly-detection-with-robust-deep-auto-encoders  
https://www.kaggle.com/mlg-ulb/creditcardfraud  
