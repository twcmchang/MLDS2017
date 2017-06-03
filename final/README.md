# Final project

### 1. BN vs. Activation functions
batch normalization 搭配不同的 activation functions，如：sigmoid, ReLU, LeakyReLU，都會有更好的 performance 嗎？還是在使用特定的 activation functions 的情況下加入 batch normalization 才會 improve 呢？

### 2. BN vs. Optimizer
加入 batch normalization 對 performance 的提升，會受到不同 optimizers (如：Adam, RMSprop, Adagrad) 的影響嗎？

### 3. BN vs. task
提出 batch normalization 的 paper 是實驗在 image 相關的 task，那 batch normalization 在非影像的 task，會有一樣的效果嗎？

### 4. BN vs. error surface
為什麼 batch normalization 會使得訓練效果比較好？宏毅哥昨天 (6/2) 說：可能是加入 batch normalization 之後，error surface 會變得比較平緩。會是神猜測嗎？

### 5. BN: layer Jacobians close to 1?
batch normalization 的作者宣稱 batch normalization 會讓 layer Jacobians 的
singular values 都接近 1，此性質對 training 會有幫助。這是真的嗎？會有什麼好處呢？

### 6. BN vs. batch size
Batch normalization 的作者有提到 batch size 太小的時候，batch normalization 表現可能會不好，該怎麼設定 batch size 才能避免這個問題呢？多大的 batch size 才足夠呢？
- 跑實驗

### 7. BN vs. training/testing distribution
當 training/testing dataset 的 distribution 差異太大時，batch normalization 表現可能會不好，那 training/testing dataset 的 distribution 差異該控制在多少的範圍內，使用 batch normalization 才會有幫助呢？
- 只拿 0 1 的資料，產生 imbalanced dataset。在 training 時作 data balancing，但在 testing 時使用原本的 distribution。看多 imbalanced 會爆炸。

### 8. BN vs. regularization (dropout, L1&L2)
Batch normalization 作者宣稱使用 batch normalization 即有 regularization 的效果，可以減少(甚至移除) dropout 或 L1, L2 regularizer 的強度。那雙管齊下會更好嗎？還是只能擇一？
- (first) 實驗有加跟沒有加，加入 dropout 0.0 - 0.5

### 9. BRN vs. Hyper parameters
Batch renomalization 被提出用來解決當 batch size 過小或 non-i.i.d batch 的問題，在 batch renormalization 的方法中，有許多的 hyper-parameters，如：moving average 的 update rate，這些 hyper-parameters 如何影響 batch renormalization 的表現呢？
- alpha, updating rate
- 用 BN 先 train N 個 batch steps 再用 BRN 會比較好 (paper: N=5000 steps)
- (optional) rmax&dmax selection.

### 10. BN/BRN vs. moving average methods
batch normalization/renormalization 在訓練過程中，會以 moving average 的方式不斷更新 testing 時使用的 mean 和 variance，有更好的方法 (如：其他的 average 方式) 去估測 testing 使用的 mean 和 variance 嗎？

# Timeline
- 1st stage: submitted before 6/10, 23:59:59
- 2nd stage: submitted before 6/23, 23:59:59
