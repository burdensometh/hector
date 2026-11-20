hector
======

Golang machine learning lib. Currently, it can be used to solve binary classification problems.

# Supported Algorithms

1. Logistic Regression
2. Factorized Machine
3. CART, Random Forest, Random Decision Tree, Gradient Boosting Decision Tree
4. Neural Network

# Dataset Format

Hector support libsvm-like data format. Following is an sample dataset

	1 	1:0.7 3:0.1 9:0.4
	0	2:0.3 4:0.9 7:0.5
	0	2:0.7 5:0.3
	...

# How to Run

## Run as tools

hector-cv.go will help you test one algorithm by cross validation in some dataset, you can run it by following steps:

	go get github.com/xlvector/hector
	go install github.com/xlvector/hector/hectorcv
	hectorcv --method [Method] --train [Data Path] --cv 10

Here, Method include

1. lr : logistic regression with SGD and L2 regularization.
2. ftrl : FTRL-proximal logistic regreesion with L1 regularization. Please review this paper for more details "Ad Click Prediction: a View from the Trenches".
3. ep : bayesian logistic regression with expectation propagation. Please review this paper for more details "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine"
4. fm : factorization machine
5. cart : classifiaction tree
6. cart-regression : regression tree
7. rf : random forest
8. rdt : random decision trees
9. gbdt : gradient boosting decisio tree
10. linear-svm : linear svm with L1 regularization
11. svm : svm optimizaed by SMO (current, its linear svm)
12. l1vm : vector machine with L1 regularization by RBF kernel
13. knn : k-nearest neighbor classification

hector-run.go will help you train one algorithm on train dataset and test it on test dataset, you can run it by following steps:

	cd src
	go build hector-run.go
	./hector-run --method [Method] --train [Data Path] --test [Data Path]

Above methods will direct train algorithm on train dataset and then test on test dataset. If you want to train algorithm and get the model file, you can run it by following steps:

	./hector-run --method [Method] --action train --train [Data Path] --model [Model Path]

Then, you can use model file to test any test dataset:

	./hector-run --method [Method] --action test --test [Data Path] --model [Model Path]

# Resumable FTRL Training with Checkpoints

The `ni`/`zi` model file above only holds prediction weights. It is enough to
load a model for inference, but it is NOT enough to resume optimization: a
resume also needs the hyper-parameters, the current epoch and the position of
the next source record. For that, FTRL supports durable batched checkpointing.

Start (or restart) a resumable training run with `--checkpoint`:

	./hector-run --method ftrl --action train --train [Data Path] \
		--checkpoint [Checkpoint Path] --batch-size 1000 \
		--model [Model Path]

Semantics:

1. Samples are processed in fixed batches. A batch is only *confirmed* after
   every one of its samples has been updated and the checkpoint has been
   atomically written (temp file + fsync + rename). A crash inside a batch
   simply replays the whole batch on restart.
2. The checkpoint stores the full optimizer state (`ni`, `zi`), all
   hyper-parameters, the epoch, the next source-record position, the batch
   size, an SHA-256 fingerprint of the input file and a fingerprint of the
   training definition (method, hyper-parameters, batch size).
3. On restart the fingerprints are verified. A checkpoint is **rejected** (the
   process errors out without touching model state) when the input content or
   the training definition has changed.
4. If a checkpoint write fails, the previous complete checkpoint file is
   retained; the durable resume point rolls back to the last confirmed batch.
5. Replaying only the unconfirmed batches yields exactly the parameters an
   uninterrupted run would have produced, because FTRL updates stay sequential
   and deterministic in sample order.
6. The `--model` file remains an inference-only artifact. Loading it back gives
   prediction weights but cannot masquerade as a resumed optimizer; resuming the
   optimization process always requires the checkpoint.
7. Without `--checkpoint`, FTRL training behaves exactly as before.

# Benchmark

## Binary Classification

Following are datasets used in benchmarks, You can find them from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/)

1. heart
2. fourclass

I will do 5-fold cross validation on the dataset, and use AUC as evaluation metric. Following are the results:

DataSet | Method | AUC
------- | ------ | ---
heart   | FTRL-LR   |0.9109
heart   | EP-LR | 0.8982
heart | CART | 0.8231
heart | RDT | 0.9155
heart | RF | 0.9019
heart | GBDT | 0.9061
fourclass | FTRL-LR | 0.8281
fourclass | EP-LR | 0.7986
fourclass | CART | 0.9832
fourclass | RDT | 0.9925
fourclass | RF | 0.9947
fourclass | GBDT | 0.9958

