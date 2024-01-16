# Lecture 1

- Explain why we need separate train and test data? What is generalization and how the concept relates to underfitting and overfitting? [10]
    - **Separate train / test data** - to be able to evaluate the model on unseen data, estimate the generalization error and test the model on data that was not used for training - to see how it's going to perform in the real world.

- Define prediction function of a linear regression model and write down L2-regularized mean squared error loss. [10]
    - **Prediction function** - linear function of the input variables, i.e. $$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_Dx_D$$ Sometimes the bias is separate from the weights, i.e. $$\hat{y} = b + w_1x_1 + w_2x_2 + ... + w_Dx_D$$

    - **L2-regularized mean squared error loss**: $$L(w) = \frac{1}{2}\lvert\lvert Xw - t \rvert\rvert^2 + \frac{\lambda}{2}\lvert\lvert w \rvert\rvert^2$$

- Starting from unregularized sum of squares error of a linear regression model, show how the explicit solution can be obtained, assuming XTXXTX is regular. [20]
    - **Unregularized sum of squares error**: $$L(w) = \frac{1}{2}\lvert\lvert Xw - t \rvert\rvert^2$$
    $$\frac{1}{2}\sum_i^N(x_i^T w - t_i)^2 \text{- we want to minimize this}$$
    $$\left(\frac{1}{2}\sum_i^N(x_i^T w - t_i)^2\right)' = 0 \text{ (differentiate w.r.t. w)}$$ 
    $$\frac{1}{2}\sum_i^N 2 (x_i^T w - t_i)x_{ij} = 0$$
    $$\sum_i^N (x_i^T w - t_i)x_{ij} = 0$$
    $$X_{\star, j}^T (X w - t) = 0$$
    $$X^T (X w - t) = 0$$
    $$X^T X w - X^T t = 0$$
    $$w = (X^T X)^{-1}X^T t$$

# Lecture 2

- Describe standard gradient descent and compare it to stochastic (i.e., online) gradient descent and minibatch stochastic gradient descent. [10]
    - in the standard one, we compute the gradient using all the training data
    - in the stochastic one, we compute the gradient using only one example sampled from the training data
    - minibatch SGD - we compute the gradient using a small subset of the training data sampled from the training data
    
- Write an L2-regularized minibatch SGD algorithm for training a linear regression model, including the explicit formulas of the loss function and its gradient. [10]
    - loss function
    $$ E(w) = \mathbb{E}_{\hat p \text{ data}} ( \frac{1}{2} (xw - t)^2) + \frac{\lambda}{2} \lvert\lvert w \rvert\rvert^2$$
    - estimation for a minibatch: 
    $$ E(w) = \frac{1}{2\lvert B \rvert} \sum_{i \in B} (x_i w - t_i)^2 + \frac{\lambda}{2} \lvert\lvert w \rvert\rvert^2$$
    - gradient for a minibatch - derivative of the loss function w.r.t. $w$:
    $$ \frac{\partial E(w)}{\partial w} = \frac{1}{\lvert B \rvert} \sum_{i \in B} (x_i w - t_i) x_i + \lambda w$$

- Does the SGD algorithm for linear regression always find the best solution on the training data? If yes, explain under what conditions it happens, if not explain why it is not guaranteed to converge. [20]
    - Not always - if the loss function is convex and continuous, SGD converges to the unique optimum only if:
        - all the learning rates are positive
        - $\sum \alpha_i = \infty$
        - $\sum \alpha_i^2 < \infty$, i.e. the rates are converging to zero
    - if the loss function is not convex, SGD only converges to a local optimum

- After training a model with SGD, you ended up with a low training error and a high test error. Using the learning curves, explain what might have happened and what steps you might take to prevent this from happening. [10]
    - Low training error + high test error = overfitting on the training data. We need to regularize the weights, or stop the training earlier (take the best validation checkpoint).

- You were provided with a fixed training set and a fixed test set and you are supposed to report model performance on that test set. You need to decide what hyperparameters to use. How will you proceed and why? [5]
    - todo

- What method can be used for normalizing feature values? Explain why it is useful. [5]
    - **Normalization** to fit all the values in $[0,1]$ range (subtract the feature minimum, divide by maximum - minimum)
    - **Standardization** to center the mean to $0$ and have unit variance

    - Features of different scales would need different learning rates.

# Lecture 3

- Define binary classification, write down the perceptron algorithm and show how a prediction is made for a given example. [10]
    - **Binary classification** - classification into two classes, e.g. deciding whether an email is spam or not, whether a patient has a disease or not, etc.
        - can be modelled as a linear regression with a threshold function - e.g. if the output is above the threshold, the example is classified as positive, otherwise as negative
        - this threshold is usually set to $0$ $\rightarrow$ symmetry, bias can be trained away etc.
        - points with $0$ output are on the **decision boundary**
    - **Perceptron algorithm** - a simple algorithm for learning a linear classifier
        - targets are $t_i \in \{-1, 1\}$
        - in the beginning, we initialize the weights to $0$
        - for each example $x_i$:
            - if $w^T x_i > 0$ and $t_i = 1$ or $w^T x_i < 0$ and $t_i = -1$, we do nothing
            - otherwise, we update the weights: $w \leftarrow w + t_i x_i$
        - the algorithm converges only if the data is **linearly separable**

- For discrete random variables, define entropy, cross-entropy, Kullback-Leibler divergence, and prove the Gibbs inequality (i.e., that KL divergence is non-negative). [20]
    - **Entropy** - measure of uncertainty of a random variable, $$H(P) = \mathbb{E}_{x ~ P}(I(x)) = \mathbb{E} (-\log p(x_i)) = \text{(for discrete r.v.)} = - \sum_{i=1}^n p(x_i) \log p(x_i)$$
    - **Cross-entropy** - shows how much two probability distributions "differ" $$H(P, Q) = \mathbb{E}_{x ~ P}(-\log q(x)) = \text{(for discrete r.v.)} = - \sum_{i=1}^n p(x_i) \log q(x_i)$$
    - **Gibbs inequality** $$H(P, Q) \geq H(P)$$
    $$H(P,Q) = H(P) \Leftrightarrow P = Q$$
    - **GI Proof:**
        $$H(P) - H(P,Q) \leq 0$$
        $$H(P) - H(P,Q) = \sum P(x) \log\left(\frac{Q(x)}{P(x)}\right)$$
        $$\text{we use the logarithm inequality here: } (log(x) \leq x - 1)$$
        $$\sum P(x) \log\left(\frac{Q(x)}{P(x)}\right) \leq \sum P(x) \left(\frac{Q(x)}{P(x)} - 1\right)$$
        $$\sum P(x) \log\left(\frac{Q(x)}{P(x)}\right) \leq \sum Q(x) - P(x)$$
        $$\sum P(x) \log\left(\frac{Q(x)}{P(x)}\right) \leq \sum Q(x) - \sum P(x)$$
        $$\text{both P and Q are probability distributions, they sum up to 1}$$
        $$\sum P(x) \log\left(\frac{Q(x)}{P(x)}\right) \leq 0 \space\space\space\square$$
    - **KL divergence**
        $$D_{KL}(P||Q) = H(P,Q) - H(P)$$
        
- Explain the notion of likelihood in maximum likelihood estimation. [5]
    - We are trying to train a model that returns a distribution on the data values.
    - We have a dataset of $N$ examples $(x_i)$.
    - Now, what are the parameters of the model that maximize the probability of the data (that the data were sampled from this distribution)?
        - this "probability of the data" is called **likelihood**

- Describe maximum likelihood estimation, as minimizing NLL, cross-entropy, and KL divergence. [20]
    $$argmax_{w} p_{\text{model}}L(X,w) = \prod p_{\text{model}}(x_i, w) $$
    $$argmax_{w} p_{\text{model}}L(X,w) = \sum \log p_{\text{model}}(x_i, w) $$
    $$argmin_{w} p_{\text{model}}L(X,w) = \sum - \log p_{\text{model}}(x_i, w) $$
    $$argmin_{w} p_{\text{model}}L(X,w) = \mathbb{E}_{x \sim p_{\text{data}}} - \log p_{\text{model}}(x, w) $$
    $$argmin_{w} p_{\text{model}}L(X,w) = H(p_{\text{data}}, p_{\text{model}}) + \text{const}$$
    $$argmin_{w} p_{\text{model}}L(X,w) = D_{KL}(p_{\text{data}}||p_{\text{model}}) + \text{const}$$

- Considering binary logistic regression model, write down its parameters (including their size) and explain how prediction is performed (including the formula for the sigmoid function). Describe how we can interpret the outputs of the linear part of the model as logits. [10]
    -  Parameters - weight vector $w$ and bias $b$
    -  The prediction is performed using the sigmoid function:
    $$\hat{y} = \sigma(w^T x + b)$$
    - We receive a probability of the example belonging to the positive class.
    - The output of the linear part of the model is called **logits**.
    $$
    p(C_1 \vert x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-w^T x - b}}
    $$
    $$
    p(C_1 \vert x)({1 + e^{-w^T x - b}}) = 1
    $$
    $$
    p(C_1 \vert x) e^{-w^T x - b}  = 1 - p(C_1 \vert x)
    $$
    $$
    e^{-w^T x - b}  = \frac{1 - p(C_1 \vert x)}{p(C_1 \vert x)}
    $$
    $$
    w^T x + b = \log \frac{p(C_1 \vert x)}{p(C_0 \vert x)}
    $$

- Write down an $L2$-regularized minibatch SGD algorithm for training a binary logistic regression model, including the explicit formulas of the loss function and its gradient. [20]
    - The loss function is the negative log likelihood:
    $$E(w) = \frac{1}{N} \sum_i - log(C_{ti} | x_i , w) + \frac{\lambda}{2} \lvert\lvert w \rvert\rvert^2$$
    - The gradient is then
    $$\frac{\partial E(w)}{\partial w} = \frac{1}{N} \sum_i (y_i - t_i) x_i + \lambda w$$

# Lecture 4

- Define mean squared error and show how it can be derived using MLE. [10]
    - **Mean squared error** - $$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - t_i)^2$$
    - Deriving from MLE - we're trying to find the parameters for the Normal distribution that maximize the probability of the data (likelihood):
    $$
    \argmax_{w} p(t \vert X, w) = \argmin_{w} \sum_i - \log( p(t_i \vert X_i, w) )
    $$
    Now we assume that the data is normally distributed with mean $y_i$ and variance $\sigma^2$ - we want to know the probability of the actual target $t_i$ given the mean $y_i$ and variance $\sigma^2$:
    $$
    p(t_i \vert y_i, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(t_i - y_i)^2}{2 \sigma^2}}
    $$
    $$
    \argmin_{w} \sum_i - \log( p(t_i \vert X_i, w) ) = \argmin_{w} \sum_i - \log \left( \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(t_i - y_i)^2}{2 \sigma^2}} \right)
    $$

    The logarithm and the fraction cancel out, the multiplicative constant is irrelevant for the optimization, so we can simplify the expression to:
    $$
    \argmin_{w} \sum_i \frac{(t_i - y_i)^2}{2 \sigma^2}
    $$

    This is the mean squared error, so we can see that the mean squared error is the negative log likelihood of the data under the assumption that the data is normally distributed with mean $y_i$ and variance $\sigma^2$.

- Considering $K$-class logistic regression model, write down its parameters (including their size) and explain how prediction is performed (including the formula for the softmax function). Describe how we can interpret the outputs of the linear part of the model as logits. [10]
    - Matrix of weights $W$ of size $D \times K$ and vector of biases $b$ of size $K$
    - After getting the $K$ outputs of the linear part of the model, we apply the softmax function to get the probabilities of the example belonging to each of the classes:
    $$
    \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
    $$
    If we pin one of the outputs to $0$, the linear part of the prediction is odds of the picked class compared to the pinned output.
    
- Explain the relationship between the sigmoid function and softmax. [5]
    - sigmoid is a special case of softmax on $[0, x]$

- Write down an $L2$-regularized minibatch SGD algorithm for training a $K$-class logistic regression model, including the explicit formulas of the loss function and its gradient. [20]
    - SGD as always (iterative, update the weights after each minibatch)
    - Loss function:
    $$
    E(W) = \frac{1}{N} \sum_i - \log p(C_{ti} \vert x_i, W) + \frac{\lambda}{2} \lvert\lvert W \rvert\rvert^2
    $$
    - Gradient:
    $$
    \frac{\partial E(W)}{\partial W} = \frac{1}{N} \sum_i (y_i - \mathbf{1}_{t_{i}}) x_i + \lambda W
    $$

- Prove that decision regions of a multiclass logistic regression are convex. [10]
    - Convex - for any two points from a decision region, all the points on the line connecting them are also in the decision region.
        - We can show this from the linearity of the matrix multiplication. The prediction of the midpoint is the midpoint of the predictions of the two points.
        $$x = \lambda x_a + (1-\lambda) x_b$$
        $$y(x) = \lambda y(x_a) + (1-\lambda) y(x_b)$$
        - The decision region is then the set of points where the prediction is the highest, which is the same for $x_a$ and $x_b$.
        - The summation on the distribution is linear, so for the midpoint, the highest probability is on the same class as for $x_a$ and $x_b$.

- Considering a single-layer MLP with $D$ input neurons, $H$ hidden neurons, $K$ output neurons, hidden activation $f$, and output activation $a$, list its parameters (including their size) and write down how the output is computed. [10]
    - Parameters:
        - weight matrix $W_1$ of size $D \times H$
        - bias vector $b_1$ of size $H$
        - weight matrix $W_2$ of size $H \times K$
        - bias vector $b_2$ of size $K$
    - The value is computed as follows:
        - $h = f(W_1^T x + b_1)$
        - $y = a(W_2^T h + b_2)$

- List the definitions of frequently used MLP output layer activations (the ones producing parameters of a Bernoulli distribution and a categorical distribution). Then write down three commonly used hidden layer activations (sigmoid, tanh, ReLU). [10]
    - Bernoulli distribution:
        - sigmoid - $a(x) = \frac{1}{1 + e^{-x}}$
    - Categorical distribution:
        - softmax - $a(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$
    - Hidden layer activations:
        - sigmoid - $\sigma(x) = \frac{1}{1 + e^{-x}}$
        - tanh - $f(x) = 2 \sigma(2x) - 1$
        - ReLU - $f(x) = \max(0, x)$

# Lecture 5

- Considering a single-layer MLP with $D$ input neurons, a ReLU hidden layer with $H$ units and a softmax output layer with $K$ units, write down the explicit formulas of the gradient of all the MLP parameters (two weight matrices and two bias vectors), assuming input $x$, target $t$, and negative log likelihood loss. [20]

    - Loss function:
    $$E(W) = - \log p(C_t \vert x, W)$$
    - Gradient:
    $$\frac{\partial E(L)}{\partial y_in} = y_i - \mathbf{1}_{t_i}$$
    $$\frac{\partial E(L)}{\partial b_2} = y_i - \mathbf{1}_{t_i}$$
    $$\frac{\partial E(L)}{\partial W_2} = (y_i - \mathbf{1}_{t_i}) h_i^T$$
    $$\frac{\partial E(L)}{\partial h_i} = W_2 (y_i - \mathbf{1}_{t_i})$$
    $$\frac{\partial E(L)}{\partial b_1} = \frac{\partial E(L)}{\partial h_i} \odot \mathbf{1}_{h_i > 0}$$
    $$\frac{\partial E(L)}{\partial W_1} = \frac{\partial E(L)}{\partial h_i} \odot \mathbf{1}_{h_i > 0} x_i^T$$

- Formulate the Universal approximation theorem. [10]
    - For a given $\varepsilon$ and any continuous function $f: \left[0,1\right]^D \rightarrow R$, there exists a single-hidden-layer MLP with $H \in \mathbb{N}$ hidden units, such that the maximum absolute error of the MLP on the given subset is less than $\varepsilon$.

- How do we search for a minimum of a function $f(x):R^D \rightarrow R$ subject to equality constraints $g_1(x)=0,\dots,g_m(x)=0 $? [10]
    - We use Lagrangian, i.e. 
    $$L(x, \lambda) = f(x) - \sum_{i=1}^m \lambda_i g_i(x)$$
    - there exist $\lambda_i$ Lagrange multipliers for each constraint so that the gradient of the Lagrangian for the constrained extreme is $0$.

- Prove which categorical distribution with NN classes has maximum entropy. [10]
    - Maximizing $H(p)$ subject to $\sum_{i=1}^N p_i = 1$ and $p_i \geq 0$ for all $i$.
    - We form the Lagrangian:
    $$L(p, \lambda) = - \sum_{i=1}^N p_i \log p_i - \lambda \left(\sum_{i=1}^N p_i - 1\right)$$
    - We differentiate w.r.t. $p_i$ and set the derivatives to $0$:
    $$\frac{\partial L}{\partial p_i} = \log p_i + 1 - \lambda = 0$$
    $$p_i = e^{\lambda - 1}$$
    - We use the second constraint to show that the solution is uniform distribution:
    $$\sum_{i=1}^N p_i = \sum_{i=1}^N e^{\lambda - 1} = e^{\lambda - 1} \sum_{i=1}^N 1 = e^{\lambda - 1} N = 1$$
    $$e^{\lambda - 1} = \frac{1}{N}$$

- Consider derivation of softmax using maximum entropy principle, assuming we have a dataset of N examples $(x_i,t_i),x_i \in R^D, t_i \in \{ 1,2,\dots,K \}$. Formulate the three conditions we impose on the searched $\pi: R^D \rightarrow R^K$, and write down the Lagrangian to be minimized. [20]
    - For given data, we want to find the distribution $\pi$ that models the distribution of classes for a given input $x$.
        - We have following conditions:
            - $\sum_{i=1}^K \pi(x)_i = 1$ - the probabilities sum up to $1$ (for each $x$)
            - $\sum_{i=1}^K \pi_i(x) t_i = \mathbb{E}_{t|x} t = \hat{t}$ - the expected value of the target is $\hat{t}$
            - $\text{for  } 0 \leq i \leq K \text{ we have } \pi_i(x) \geq 0$ - the probabilities are non-negative
        - We are optimizing the following Lagrangian:
            $$\sum_{i=1}^{N}\sum_{k=1}^{K} \pi(x_i)_k \log \pi(x_i)_k + \lambda_1 \left(\sum_{k=1}^{K} \pi(x_i)_k - 1\right) + \lambda_2 \left(\sum_{k=1}^{K} \pi(x_i)_k t_i - \hat{t}\right) + \sum_{i=1}^{N}\sum_{k=1}^{K} \lambda_3 \pi(x_i)_k$$

- Define precision (including true positives and others), recall, $F1$ score, and $F\beta$ score (we stated several formulations for $F1$ and $F\beta$ scores; any one of them will do). [10]
    - **Precision** - how many of the predicted positives are actually positive ($\frac{TP}{TP + FP}$)
    - **Recall** - how many of the actual positives were predicted as positive ($\frac{TP}{TP + FN}$)
    - **$F1$ score** - harmonic mean of precision and recall ($\frac{2}{\frac{1}{precision} + \frac{1}{recall}}$)
    - **$F\beta$ score** - weighted harmonic mean of precision and recall ($\frac{1 + \beta^2}{\frac{1}{precision} + \beta^2\frac{1}{recall}}$)

- Explain the difference between micro-averaged and macro-averaged $F1$ scores. [10]
    - In multi-class classification, we can compute the $F1$ scores for each class separately and then average them
        - This is called **macro-averaging**.
    - Or we compute the true positives, false positives and false negatives for each class separately and then compute the $F1$ score from their sums.
        - This is called **micro-averaging**.

- Explain (using examples) why accuracy is not a suitable metric for unbalanced target classes, e.g., for a diagnostic test for a contagious disease. [5]
    - Imbalanced Dataset:
        - Infected (Positive Class): 10 instances
        - Healthy (Negative Class): 990 instances

    - Confusion Matrix:
        - True Positives (TP): 5
        - True Negatives (TN): 980
        - False Positives (FP): 10
        - False Negatives (FN): 5

    - Accuracy: $\frac{TP + TN}{TP + TN + FP + FN} = \frac{5 + 980}{5 + 980 + 10 + 5} = 0.985$

# Lecture 6

- Explain how is the TF-IDF weight of a given document-term pair computed. [5]
    - **TF-IDF**:
        - **TF** - term frequency, meaning how many times does the term appear in the document (normalized by the document length)
        $$tf(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
        - **IDF** - inverse document frequency, meaning how rare is the term in the corpus
        $$idf(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|}$$
        - **TF-IDF** - $$tfidf(t,d,D) = tf(t,d) \cdot idf(t,D)$$

- Define conditional entropy, mutual information, write down the relation between them, and finally prove that mutual information is zero if and only if the two random variables are independent (you do not need to prove statements about DKL). [10]
    - Conditional entropy: $H(Y|X) = \mathbb{E}[\mathbb{I}(y|x)]$
        - in discrete case: $H(Y|X) = - \sum_{x,y} p(x,y) \log p(y|x)$
    - Mutual information - how many bits do we learn about $y$ if we find out the value of $x$?
        - $I(X;Y) = H(Y) - H(Y|X)$
        - $I(X;Y) = \mathbb{E}_{x,y} \left(\log\frac{P(x,y)}{P(x)P(y)}\right)$
        - $I(X;Y) = 0 \Leftrightarrow X \perp Y$, i.e. only if $P(x,y) = P(x)P(y)$ (substitute into the formula above - logarithm of $1$ is $0$)

- Show that TF-IDF terms can be considered portions of suitable mutual information. [10]
    - Let's first assume we pick the document uniformly at random from the corpus, i.e.
    $$P(d) = \frac{1}{N}$$
    $$H(D) = \log N$$
    - If we pick a term, the probability of picking a specific document from this reduced corpus is
    $$P(d|t) = \frac{1}{|\{d \in D : t \in d\}|}$$
    - A conditional entropy in this case is
    $$H(D|T = t) = I(d|t \in d) = \log |\{d \in D : t \in d\}|$$
    - The mutual information for one given term and document is then
    $$I(d;t) = H(D) - H(D|t) = \log N - \log |\{d \in D : t \in d\}| = \log \frac{N}{|\{d \in D : t \in d\}|} = IDF(t)$$
    - The mutual information for all the terms and documents is then
    $$I(D;T) = \sum_{t, d} p(d) p(t|d) I(d;t) = \frac{1}{N} \sum_{t, d} T(t, d) IDF(t)$$

- Explain the concept of word embedding in the context of MLP and how it relates to representation learning. [5]
    - Word embedding - mapping words to vectors, where similar words are close to each other in the vector space
    - Representation learning - **learning** a representation of the data instead of engineering the features - often done by training a neural network and using the hidden layer activations as the representation
    - Assume a one-hidden-layer MLP we're trying to teach some probabilities of words given the context. In the input, the words are represented as one-hot vectors. The weight matrix for the hidden layer is then the word embedding matrix - every column represents the corresponding word (one-hot encoding).

- Describe the skip-gram model trained using negative sampling. [10]
    - Skip-gram - variant of word2vec, where we try to predict the **context** given the central word (instead of predicting the word using the context)
    - For every word in the corpus, we sample a context window of size $C$ and try to predict the words in the window using the central word (we create pairs $\text{central word - context word}$). 
        - Learning a classifier that would give us a distribution on all the words in the vocabulary would be too expensive, so we rather train binary classifiers for each central-context word pair.
    - Negative sampling - in loss, we don't only consider the positive examples (words in the context window), but also sample some negative examples (words that are not in the context window).
    $$-\log \sigma (e_w^T v_c) - \sum_i^K \log \sigma(-e_w^Tv_i)$$

- How would you proceed to train a part-of-speech tagger (i.e., you want to assign each word with its part of speech) if you only could use pre-trained word embeddings and MLP classifier? [5]
    - POS tagging = sequence labeling = classification for each word in the sentence
        - we can use a sliding window of size $2C + 1$ to get the context for each word
        - we can use the word embeddings as the input to the MLP
        - we can use the MLP to predict the POS tag for each word (given the context)

# Lecture 7

- Describe $k$-nearest neighbors prediction, both for regression and classification. Define $L_p$​ norm and describe uniform, inverse, and softmax weighting. [10]
    - **$k$-nearest neighbors** - for a given example, we find the $k$ closest examples in the training set and use their targets to predict the target of the given example
        - non-parametric, no training phase (only store the labelled data)
        - **$L_p$ norm** - $L_p(x, y) = \left(\sum_i \lvert x_i - y_i \rvert^p\right)^{\frac{1}{p}}$
        - **Weighting**:
            - uniform - all the neighbors have the same weight
            - inverse - the weight is inversely proportional to the distance
            - softmax - the weight is proportional to the distance (the closest neighbor has weight $1$, the farthest has weight $0$)
        - **Regression**: weighted average of the targets of the $k$ nearest neighbors
        - **Classification**: majority vote of the $k$ nearest neighbors (can be weighted, then $\text{argmax}$ ed)

- Show that $L2$-regularization can be obtained from a suitable prior by Bayesian inference (from the MAP estimate). [10]
    - TODO

- Write down how $p(C_k∣x)$ is approximated in a Naive Bayes classifier, explicitly state the Naive Bayes assumption, and show how is the prediction performed. [10]
    - The naive bayes assumption is that the features are independent given the class 
    $$p(x|C_k) = \prod_{d=1}^D p(x_d|C_k)$$
    - The prediction is then
    $$\hat{y} = \text{argmax}_k p(C_k) \prod_{d=1}^D p(x_d|C_k)$$

- Considering a Gaussian naive Bayes, describe how are $p(x_d∣C_k)$ modeled (what distribution and which parameters does it have) and how we estimate it during fitting. [10]
    - For continuous features, we can use a Gaussian naive Bayes:
    - The $p(x_d|C_k)$ is modelled as a normal distribution with mean $\mu_{kd}$ and variance $\sigma_{kd}^2$ - we try to estimate these parameters during fitting.
    - For a given class and a given feature, we estimate the mean and variance as the sample mean and variance of the feature values of the examples in the class.
        - e.g. for a single pixel of the handwritten digit $3$ during MNIST, mean is the average value of the pixel in all the examples of $3$, variance is the variance of the pixel value in all the examples of $3$.

- Considering a Bernoulli naive Bayes, describe how are $p(x_d∣C_k)$ modeled (what distribution and which parameters does it have) and how we estimate it during fitting. [10]
    - For binary features, we can use a Bernoulli naive Bayes:
    $$p(x_d|C_k) = p_{kd}^{x_d} (1 - p_{kd})^{1 - x_d}$$
    - We can optimize the parameters $p_{kd}$ using MLE over the training set - with differentiation, we arrive at
    $$p_{kd} = \frac{1}{N_k} \sum_{i=1}^N x_{id}$$
    - to "soften" the strictness of the classifier, we can add "Laplace" or "additive" smoothing - add $\alpha$ and $2\alpha$ to the nominator / denominator respectively.

# Lecture 8

- Prove that independent discrete random variables are uncorrelated. [10]
    - definition: $Cov(x,y) = \mathbb{E} \left((x - \mathbb{E}(x)) (y - \mathbb{E}(y)) \right)$
    - if $x$ and $y$ are independent, then $P(x,y) = P(x)P(y)$, which means that the covariance is $0$.

- Write down the definition of covariance and Pearson correlation coefficient $\rho$, including its range. [10]
    - Covariance: 
    $$Cov(x,y) = \mathbb{E} \left((x - \mathbb{E}(x)) (y - \mathbb{E}(y)) \right)$$
    - Pearson correlation coefficient: 
        $$\rho = \frac{Cov(x,y)}{\sqrt{Var(x)} \sqrt{Var(y)}}$$
        - ranges from $-1$ to $1$
        - measures the linear correlation between two variables

- Explain how are the Spearman's rank correlation coefficient and the Kendall rank correlation coefficient computed (no need to describe the Pearson correlation coefficient). [10]
    - Spearman's rank correlation coefficient
        - used to measure non-linear correlation between two variables
        - it's the Pearson's correlation coefficient measured on orderings of the original data (see the [IQ vs TV hours spent](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient#Example) on Wikipedia)

    - Kendall's rank correlation coefficient
        $$\frac{\text{\# of concordant pairs} -\text{\# of discordant pairs}}{n \choose 2 }$$
        - if all the pairs are concordant, the coefficient is $1$
        - if all the pairs are discordant, the coefficient is $-1$

- Describe setups where a correlation coefficient might be a good evaluation metric. [5]
    - Ranking (e.g. search engine results)
    - Pair similarity (e.g. word embeddings vs linguistic experiments - scores for pairs of words)
        - what's the correlation between the human-annotated scores and the embedding distances?

- Define Cohen's $\kappa$ and explain what it is used for when preparing data for machine learning. [10]
    - Determining the inter-rater agreement for qualitative (categorical) items.
    $$ \kappa = \frac{p_o - p_e}{1 - p_e}$$
    - where $p_0$ is the proportion of data points where both raters agreed on the label and $p_e$ is the probability that the raters agree by chance.

- Considering an averaging ensemble of $M$ models, prove the relation between the average mean squared error of the ensemble and the average error of the individual models, assuming the model errors have zero means and are uncorrelated. [20]
   - We define the mean square error of the i-th model as $\mathbb{E}\left[ \varepsilon_i^2 (x) \right]$ 
   - Now we can define the mean square error of the ensemble as
        $$
        MSE_\text{ensemble} = \mathbb{E}\left[ \left( \frac{1}{M} \sum_{i=1}^M \varepsilon_i(x) \right)^2 \right]
        $$
        - this comes from the way we calculate the predictions, i.e. we average the predictions of the individual models
   - If the errors have zero mean and are uncorrelated, it means
   $$
   \forall i, j : \mathbb{E}\left[ (\varepsilon_i(x) - \mathbb{E}(\varepsilon_i(x) )) (\varepsilon_j(x) - \mathbb{E}(\varepsilon_j(x) )) \right] = 0
   $$
   $$
   \forall i, j : \mathbb{E}\left[ \varepsilon_i(x) \varepsilon_j(x) \right] = 0
   $$
   - So we can continue with the MSE of the ensemble:
   $$
    MSE_\text{ensemble} = \mathbb{E}\left[ \left( \frac{1}{M} \sum_{i=1}^M \varepsilon_i(x) \right)^2 \right]
   $$
   $$
    MSE_\text{ensemble} = \mathbb{E}\left[ \frac{1}{M^2} \sum_{i=1}^M \varepsilon_i(x)^2 \right] \text{ - all the other $\varepsilon_i(x)$ are 0}
   $$
   - the MSE of the ensemble is then $\frac{1}{M}$ - times the average MSE of the individual models.

- Explain knowledge distillation: what it is used for, describe how it is done. [10]
    - Knowledge distillation is a method for training a smaller model to mimic the predictions of a larger model.
    - We train the larger model on the training set, receiving the pseudolikelihood distribution on the output data.
        - Then we train the smaller model, trying to minimize the mutual entropy between the predictions of the larger model (the entire distribution of probabilities on classes) and the smaller model.

# Lecture 9

- In a regression decision tree, state what values are kept in internal nodes, define the squared error criterion and describe how is a leaf split during training (without discussing splitting constraints). [10]
    - **Regression decision tree** - a decision tree for regression, i.e. the output is a real number
    - **Internal nodes** - split points, i.e. the values of the features that are used to split the data
    - **Leaf nodes** - the data points are assigned to the leaf nodes, the output is the mean of the target values of the data points in the leaf node
        - when we train the tree, we keep the mean of the target values of the data points in the leaf node
        - during split of a leaf node, we go through all the features and all the values of the features and try to find the split on a feature that minimizes the splitting criterion, meaning 
        $$ c(L) + c(R) - c(Root) $$ 
        is the lowest. The criterion for a node is computed as a sum of square differences of the target values of the data points in the node and the mean of the target values of the data points in the node.

- In a $K$-class classification decision tree, state what values are kept in internal nodes, define the Gini index and describe how is a node split during training (without discussing splitting constraints). [10]
    - The internal nodes are split points, i.e. the values of the features that are used to split the data. The leaves are the data points, the output is the most frequent class in the leaf.
    - Gini-index:
        $$c(T) = \lvert I_T\rvert \sum_{k}p_T(k) (1 - p_T(k))$$
        - if we pick a value on random, what is the probability that it's not in the same class as the majority of the values in the node?
    - Going through all the features and all the values of the features, we try to find the split on a feature that minimizes the splitting criterion, meaning 
        $$ c(L) + c(R) - c(Root) $$ 
        is the lowest. 

- In a $K$-class classification decision tree, state what values are kept in internal nodes, define the entropy criterion and describe how is a node split during training (without discussing splitting constraints). [10]
    - The internal nodes are split points, i.e. the values of the features that are used to split the data. The leaves are the data points, the output is the most frequent class in the leaf.
    - Entropy criterion:
        $$c(T) = - \lvert I_T\rvert \sum_{k}p_T(k) log(p_T(k))$$
        - if we pick a value on random, what is the probability that it's not in the same class as the majority of the values in the node?
    - Going through all the features and all the values of the features, we try to find the split on a feature that minimizes the splitting criterion, meaning 
        $$ c(L) + c(R) - c(Root) $$ 
        is the lowest. 

- For binary classification, derive the Gini index from a squared error loss. [20]
    - Be $n_t(0)$ the number of examples with target $0$ in the node $t$ 
    - $n_t(1)$ the number of examples with target $1$ in the node $t$.
    - Let's set $p_t = \frac{n_t(1)}{n_t(0) + n_t(1)}$ - the average value in the node $t$.
        - Now, we're trying to minimize the 
        $$L(p) = \sum_{t \in T} (p_t - t_i)^2$$
    We split the cases based on the target value:
    $$L(p) = \sum_{t \in T} (p_t - t_i)^2 = n_0(p_T - 0)^2 + n_1(p_T - 1)^2$$
    todo

- For K-class classification, derive the entropy criterion from a non-averaged NLL loss. [20]
    - NLL loss = $L(p) = \sum_i -\log(p_{ti})$
        - we differentiate this, set the Lagrangian with $\sum_i p_{ti} = 1$ and get $p_{ti}$ = empirical distribution from the data
        - we substitute this into the NLL loss and get the entropy criterion
    $$L(p) = \sum_i -\log(p_{ti})$$
    $$L(p) = \sum_{k} n(k) -\log(p_t(k))$$
    $$L(p) = - \lvert I_T \rvert\sum_{k} p_t(k) \log(p_t(k))$$
    $$L(p) = \lvert I_T \rvert H(p_T)$$

- Describe how is a random forest trained (including bagging and a random subset of features) and how is prediction performed for regression and classification. [10]
    - Random forest - an ensemble of decision trees
        - **Bagging** - we train each tree on a random subset of the training data (with replacement) - if we didn't do this, all the trees would be the same
        - **Random subset of features** - in every node, we only allow ourselves to split on a random subset of features - if we didn't do this, all the trees would be (almost) the same
    - regression - mean of the predictions of the trees
    - classification - majority vote of the predictions of the trees

# Lecture 10

- Write down the loss function which we optimize in gradient boosted decision trees during the construction of $t$-th tree. Then define $g_i$​ and $h_i$​ and show the value $w_T$ of optimal prediction in node $T$ and the criterion used during node splitting. [20]
    - We try to optimize the loss
    $$\mathbb{E}(w_t, w^{t-1}) = \sum [l(t_i, y^{i-1}(x_i, w^{t-1}) + y_t(x_i, w_t))] + \frac{1}{2} \lambda \lvert\lvert w_t \rvert\rvert^2$$

    - $g_i$ - gradient of the loss w.r.t. the prediction of the previous tree - this comes from the second order Taylor approximation of the loss
    $$g_i = \frac{\partial l(t_i, y^{i-1}(x_i))}{\partial y^{i-1}(x_i)}$$
    - $h_i$ - second derivative of the loss w.r.t. the prediction of the previous tree - this comes from the second order Taylor approximation of the loss
    $$h_i = \frac{\partial^2 l(t_i, y^{i-1}(x_i))}{\partial y^{i-1}(x_i)^2}$$

    - the optimal prediction weights in the node $T$ are
        $$w_T = \frac{\sum_{i \in T} g_i}{\sum_{i \in T} h_i + \lambda}$$
        - this stems from the "sped up" SGD - we want to move in the direction of the gradient with a variable step size (the step size is the optimal prediction weight - Newton's method of finding a root of a function - the first derivative here)
    
    - the splitting criterion for a node is

    $$
    - \frac{1}{2} \left(\frac{\sum_{i \in T_L} g_i}{\sum_{i \in T_L} h_i + \lambda}\right) + const
    $$

    and we try to find such a split that the criterion for the left and right child vs the original is the lowest.

- For a $K$-class classification, describe how to perform prediction with a gradient boosted decision tree trained for $T$ time steps (how the individual trees perform prediction and how are the $K⋅TK$ trees combined to produce the predicted categorical distribution). [10]
    - the prediction distribution is a 
    $$\text{softmax}(\sum_{t=1}^T y_t(x, w_{t1}) + \sum_{t=1}^T y_t(x, w_{t2}) + \dots + \sum_{t=1}^T y_t(x, w_{tK}))$$

- What type of data are gradient boosted decision trees good for as opposed to multilayer perceptron? Explain the intuition why it is the case. [5]
    - Gradient boosted decision trees are good for tabular data, where the features have some meaning (e.g. age, height, etc.)
    - MLP's are better for high-dimensional data, where the features don't have any meaning (e.g. pixels of an image)

# Lecture 11

- Formulate SVD decomposition of matrix $X$, describe properties of individual parts of the decomposition. Explain what the reduced version of SVD is. [10]
    - Every matrix of dimensions $m \times n$ and rank $r$ can be decomposed as
    $$X = U \Sigma V^T$$
    - Where:
        - $U \in \mathbb{R}^{m \times m}$ is an orthonormal square matrix.
        - $\Sigma$ is a diagonal matrix with non-negative real numbers on the diagonal.
        - $V \in \mathbb{R}^{n \times n}$ is an orthonormal square matrix.
    - The reduced version of SVD is when we only keep the first $r$ columns of $U$ and $V$ and the first $r$ diagonal elements of $\Sigma$ - this can be more economical to compute if $r$ is much smaller than $m$ and $n$.

- Formulate the Eckart-Young theorem. [10]
    - For a given matrix $X$ and a given rank $r$, the best rank-$r$ approximation of $X$ is given by the reduced SVD decomposition of $X$.
    $$X_k = \sigma_1 u_1 v^T_1 + \sigma_2 u_2 v^T_2 + \dots + \sigma_k u_k v^T_k$$
    $$\lvert\lvert X - X_k \rvert\rvert_F \leq \lvert\lvert X - B \rvert\rvert_F$$
    For any matrix $B$ of rank $k$. The equality holds if $B = X_k$.

- Explain how to compute the PCA of dimension M using the SVD decomposition of a data matrix X, and why it works. [10]
    - We start with the mean centered data matrix $X$, i.e. from each row, we subtract the mean of the row. This corresponds to making each feature have mean $0$.
    - We compute the SVD decomposition of $X = U \Sigma V^T$
    TODO

- Given a data matrix $X$, write down the algorithm for computing the PCA of dimension $M$ using the power iteration algorithm. [20]
    - We get the matrix $X$, compute the row means and subtract them from the rows of $X$ to get the (mean-centered) matrix $X'$.
    - We compute the covariance matrix $S = \frac{1}{N} X'^T X'$.
    - For $i = 1, \dots, M$:
        - We initialize $v_i$ to a random vector.
        - For $j = 1, \dots, T$ (or until convergence):
            - We compute $v_i = S v_i$.
            - We normalize it.
        - We set S to $S - v_i v_i^T$. (we remove the largest eigenvalue and the corresponding eigenvector), so that the next iteration finds the next largest eigenvalue and eigenvector.
    - We return $XV$, which corresponds to the transformed data ($V$ is the matrix of eigenvectors).


- Describe the K-means algorithm, including the kmeans++ initialization. [20]
    - Algorithm for clustering data into K clusters
        - every cluster is defined by it's centroid (a point in the space, same dimensionality as the data)
        - we are optimizing the sum of squared distances from the data points to **their** centroids
        - the algorithm:
            - initialize the centroids (e.g. randomly)
            - assign each data point to the closest centroid
            - update the centroids to be the mean of the assigned data points
            - repeat until convergence
        - the algorithm is not guaranteed to converge to the global optimum (finds the local one)
        - the initialization is important, e.g. **kmeans++**:
            - initialize the first centroid randomly (pick one of the data points)
            - for each centroid:
                - compute the distance of each data point to the closest centroid
                - choose the next centroid with probability proportional to the square of the distance
            - repeat until all centroids are initialized

# Lecture 12

- Considering statistical hypothesis testing, define type I errors and type II errors (in terms of the null hypothesis). Finally, define what a significance level is. [10]
    - Type I error - rejecting the null hypothesis when it is true (false positive, i.e. proven guilty when innocent)
    - Type II error - not rejecting the null hypothesis when it is false (false negative, i.e. proven innocent when guilty)
    - Significance level - error rate for the type I errors, i.e. the probability of rejecting the null hypothesis when it is true
        - To an extent, this is similar to precision vs recall - we can control the significance level (precision) by choosing the threshold.

- Explain what a test statistic and a p-value are. [10]
    - **Test statistic** - a function on the observed data, a summary of them (e.g. mean, variance, etc.)
    - **p-value** - probability of obtaining a test statistic as extreme as the one observed, assuming the null hypothesis is true (i.e. our hypothesis is that people are on average 170cm tall (and this is modelled with normal distribution), we measure the height of 100 people and compute the mean, which is 175cm - the p-value is the probability of obtaining a mean as extreme as 175cm, assuming the null hypothesis is true, e.g. the area under the bell curve from 175cm to infinity)

- Write down the steps of a statistical hypothesis test, including a definition of a p-value. [10]
    - **Statistical hypothesis test**
        - we define the null hypothesis $H_0$ and (optionally) the alternative hypothesis $H_1$
        - we choose the test statistic
        - calculate the observed value of the test statistic
        - calculate the p-value, i.e. probability that the test statistic is as extreme as the observed one, assuming the null hypothesis is true
        - if the p-value is lower than the significance level $\alpha$, we reject the null hypothesis, otherwise, we accept.

- Explain the differences between a one-sample test, two-sample test, and a paired test. [10]
    - **one-sample test** - we sample data from one distribution
    - **two-sample test** - we sample data from two distributions
    - **paired test** - we sample from two distributions, but the samples are paired - e.g. we compare the predictions of two models on the same test set

- When considering multiple comparison problem, define the family-wise error rate, and prove the Bonferroni correction, which allows limiting the family-wise error rate by a given $\alpha$. [10]
    - MCP - if we run the same test multiple times (on difrrent data), the probability of making a type I error increases (we'll eventually confirm something)
    - FWER - probability of making at least one type I error in a sequence of tests
    $$FWER = P \left( \bigcup_i p_i \leq \alpha \right)$$
    - Bonferroni correction - if we want to limit the FWER to $\alpha$, we can simply divide the significance level $\alpha$ *for one experiment* by the number of tests $m$:
        $$\alpha' = \frac{\alpha}{m}$$
        $$FWER = P \left( \bigcup_i p_i \leq \frac{\alpha}{m} \right)$$
        - using Boole's inequality, we can prove that the FWER is then limited to $\alpha$:
        $$P(\bigcup A_i) \leq \sum P(A_i) \text{(Boole's inequality)}$$
        $$FWER = P \left( \bigcup_i p_i \leq \frac{\alpha}{m} \right) \leq \sum P(p_i \leq \frac{\alpha}{m}) = \sum \frac{\alpha}{m} = \alpha$$

- For a trained model and a given test set with $N$ examples and metric $E$, write how to estimate 95% confidence intervals using **bootstrap resampling**. [10]
    - **Bootstrap resampling** - we sample $N$ examples from the test set with replacement, i.e. we can sample the same example multiple times
    - We repeat this $B$ times, i.e. we get $B$ samples of size $N$ - this way, we create "$B$" test sets.
    - We compute the metric $E$ on each of these test sets, i.e. we get $B$ values of $E$.
    - We sort these values and take the $2.5\%$ and $97.5\%$ quantiles - these are the bounds of the confidence interval.

- For two trained models and a given test set with $N$ examples and metric $E$, explain how to perform a **paired bootstrap** test that the first model is better than the other. [10]
    - the same as above, i.e. we sample the examples from the test set with replacement, but for each sample, we evaluate both models and compute the difference in the metric $E$ between the two models

- For two trained models and a given test set with $N$ examples and metric $E$, explain how to perform a **random permutation test** that the first model is better than the other with a significance level $\alpha$. [10]
    - For a given number of resamplings, we:
        - Generate a randomized test set.
        - Generate a randomized "model" - test set with randomized targets (either from the first or the second model)
        - We measure the performance of this "mixed" model and append it to the performances array.
        - We measure the performance of the first model on the randomized test set and append it to the performances array.
    - We compute the number of times the performance of the mixed model was better than the performance of the first model.

# Lecture 13

- Explain the difference between deontological and utilitarian ethics. List examples on how these theoretical frameworks can be applied in machine learning ethics. [10]
    - **Deontological** - based on rules, e.g. "do not kill", "do not lie", based on the inherent value of the action, not it's consequences
        - In ML context - "consent", "privacy", "fairness", "transparency"...
        - pros: simple, clear, easy to apply
        - cons: can be too rigid, can be too vague, disregard the consequences
    - **Utilitarian** - based on consequences, e.g. "do what brings the most happiness", "do what brings the most good"
        - In ML context - "What harm can happen: psychological, political, environmental, moral, cognitive, emotional"...
        - pros: flexible, can be applied to many situations
        - cons: can be subjective, overlooking the minority, can be hard to define the objective

- List few examples of potential ethical problems related to data collection. [5]
    - **Data collection** - data can be collected in many ways, e.g.:
        - **Web scraping** - can be illegal, can be used for manipulation
        - **Log-mining** - we can be missing the consent of the users
        - **Crowdsourcing** - we can be exploiting the workers, not well paid, monotonous work

- List few examples of potential ethical problems that can originate in model evaluation. [5]
    - The metrics being optimized can be missing some important features
        - e.g. translation fluency vs. gender bias
    - Optimizing for **precision** diminishes **recall**:
        - job applications, loan applications - we only get a few known-good applicants, but there might be more good ones
