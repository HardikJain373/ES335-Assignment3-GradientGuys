# Assignment 3: Text Generation and Machine Learning Models

## Text Generation
- Replicate the [notebook](https://nipunbatra.github.io/ml-teaching/notebooks/names.html) on the next character prediction and use it for the generation of text.
- Use one of the datasets specified below for training.
- Refer to Andrej Karpathy’s blog post on [Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Visualize the embeddings using t-SNE if using more than 2 dimensions or using a scatter plot if using 2 dimensions.
- Write a Streamlit application which asks users for an input text and then predicts the next k characters.

Datasets (first few based on Effectiveness of RNN blog post from Karpathy et al.)
- Paul Graham essays
- Wikipedia (English)
- Shakespeare
- [Maths textbook](https://github.com/stacks/stacks-project)
- Something comparable in spirit but of your choice (do confirm with TA Ayush)[5 marks]

## XOR Dataset Classification
- Learn the following models on the XOR dataset (200 training instances and 200 test instances) such that all these models achieve similar results:
  - MLP
  - MLP with L1 regularization (vary the penalty coefficient by choosing the best one using a validation dataset)
  - MLP with L2 regularization (vary the penalty coefficient by choosing the best one using a validation dataset)
  - Logistic regression models on the same data with additional features (such as x1*x2, x1^2, etc.)
- Show the decision surface and comment on the plots obtained for different models.[2 marks]

## CO2 Forecasting
- Use the [Mauna Lua CO2 dataset](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv) (monthly) for forecasting.
- Perform forecasting using an MLP and compare the results with MA (Moving Average) and ARMA (Auto Regressive Moving Average) models.
- Main setting: use previous “K” readings to predict the next “T” reading.
- Comment on why you observe such results.
- For MA or ARMA, you can use any library or implement it from scratch.[2 marks]

## MNIST Classification
- Train an MLP on the MNIST dataset (60,000 training images and 10,000 test images).
- Compare against RF and Logistic Regression models.
- Evaluate using F1-score, confusion matrix.
- Identify commonly confused digits.
- Plot the t-SNE for the output from the layer containing 20 neurons for the 10 digits on the trained MLP.
- Contrast this with the t-SNE for the same layer but for an untrained model.
- Use the trained MLP to predict on the Fashion-MNIST dataset and observe the embeddings.[3 marks]
