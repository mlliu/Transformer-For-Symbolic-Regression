# Transformer-For-Symbolic-Regression


Symbolic regression is a task that aims to discover the underlying equations from a given sample of data or observation. Due to the richness of the space of mathematical expressions, symbolic regression is generally a challenging problem. Current approaches, including genetic programming and deep neural method, usually follow a two-step procedure: first predicting the symbols of function, then approximating the constants. We modify these conventional methods by proposing an end-to-end transformer-based language model to directly predict the complete mathematical expressions. The proposed model exploits the advantages of a large-scale model and global training strategy. Through comprehensive experiments, we show that our model performs robustly, eventually reaching around 78% accuracy.

In recent years, neural network has achieved state-of-art performance in various fields in computer vision, speech recognition, natural language processing, etc. However, only a few studies investigated the capacity and application of neural networks to deal with symbolic regression problems.

Symbolic regression (i.e., function identification) is a central problem in natural science, which aims to find a model, in symbolic form, that fits a given sample of data or observations. To be more specific, for a given finite sampling of value pairs of the independent variables and the associated dependent variables, the goal of symbolic regression is to find a mathematics expression, involving both the functional form and the numerical coefficients that can provide a good fit. 

Genetic programming (GP) is currently the main conventional algorithm for symbolic regression, which was introduced based on the Darwinian evolutionary process [1]. It should be noted that, genetic algorithm, while achieving reasonable prediction accuracy, cannot improve with experience because the solutions to each problem needs to be learned from scratch.

In recent years, neural network has shown to have wide applications fields in computer vision, speech recognition, natural language processing, etc. However, in symbolic regression, these deep learning-based methods are relatively new and are developing into an active research area.  For example, transformer-based large-scale methods have been proposed in [2,3]. These methods Inherited from GP follow a two-step procedure, first predicting the symbolic expressions using a neural network, and then fitting constants through by non-linear optimizer such as BFGS. We argue that there are some shortcomings in the two-step procedure. Firstly, it may not be suited to neural network approaches in some ways, as the neural network itself is a good non-linear optimizer. In addition, merely optimizing on functional form cannot provide sufficient supervision, for example,  different instances of the same symbolic mathematical form may results in different shapes of mathematical curves. Therefore, we propose an end-to-end numerical method and symbolic prediction to implement the complete mathematical equation.

In this report, we train a transformer model over a synthetic dataset to achieve end-to-end symbolic regression. The base 10 positional encoding (p10) method is used to represent numbers as a sequence of five tokens so that they can be processed by the model. After 2 days training on a single GeForce RTX 2080 GPU, the models is shown to achieve an accuracy of 78%.