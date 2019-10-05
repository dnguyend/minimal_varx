<a href="https://colab.research.google.com/github/dnguyend/minimal_varx/blob/master/minimal_VARX.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Minimal VARX: The $\AR$-state-space approach to time series analysis.
In this project we show implications an autoregressive state-space approach to time series and linear system identification.
The traditional state-space approach expresses $\by$ a function of inputs and innovations. This is the traditional approach in control theory, but in forecasting the link between past and future observation is not explicit, making it is difficult for both model fitting and forecasting.

We advocate the $\AR$ state-space approach. It is expressed as
$$\by = \bH(\bI-\bF L)^{-1}\bG L \by + \bep$$
By a theorem of Kalman, $\VAR, \VARMA$ and their exogenous counterparts are all expressible in this form. For the $\VARX(1)$ the minimal $\AR$ state-space approach is exactly the reduced rank regression, and the estimation procedure for reduced rank regression could be generalized. We show the (negative) likelhood function could be expressed as a determinant ratio. In the reduced-rank case maximizing this likelihood function leads to canonical- correlation-analysis. In the general case we must use a numerical procedure, but both the gradient and the Hessian could be computed explicitly for MLE.

For the $\VAR(p)$ case, $\bF$ could be reduced to a Jordan matrix with $\bF^p=0$. This allows an explicit classication of possible $minimal$ state-space. This helps providing a framework for parameter reductions going beyond reduced-rank regression. While in the $p=1$ case we factor the regression matrix $\bPhi$ to a product $\bH\bG$, in the general case we distribute the ranks $d_i$ on to each power $1\leq i\leq p$. These ranks define the complexity of the model. The most expensive model in term of parameter is one with all the rank concentrated at the $\Phi_p$ parameter, the least expensive is one where $\Phi_p$ of rank $1$ (and no lower power get allocate any rank). This provides an intuitive look at complexity of state-space model.

More interestingly, we can reduce the search space further by using matrices $\bS$ that commute with $\bF$. $\bG$ could be made to satisfy two sets of orthogonal relations. Hence we can apply manifold optimization technique for high dimensional problem. It turns out the manifold in question is an interesting mathematical object by itself, a vector bundle over a flag manifold. However, there is no need to get to manifold optimization to use our model.
Please consult the notebook for a demonstration.
https://github.com/dnguyend/minimal_varx/blob/master/minimal_VARX.ipynb
For details derivation of the results please consult the paper.
