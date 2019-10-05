$\newcommand{\by}{\boldsymbol{y}}}$
$\newcommand{\R}{\mathbb{R}}$ 
$\newcommand{\nmin}{n_{\min}}$
$\newcommand{\C}{\mathbb{C}}$ $\newcommand{\Z}{\mathbb{Z}}$
$\newcommand{\V}{\mathbb{V}}$
$\newcommand{\cX}{\mathcal{X}}$
$\newcommand{\cC}{\mathcal{C}}$
$\newcommand{\cG}{\mathcal{G}}$
$\newcommand{\cH}{\mathcal{H}}$
$\newcommand{\cGO}{\mathcal{GO}}$
$\newcommand{\cF}{\mathcal{F}}$
$\newcommand{\bpi}{\boldsymbol{\pi}}$
$\newcommand{\cK}{\mathcal{K}}$
$\newcommand{\cQ}{\mathcal{Q}}$
$\newcommand{\cR}{\mathcal{R}}$
$\newcommand{\cS}{\mathcal{S}}$
$\newcommand{\fll}{\mathfrak{l}}$
$\newcommand{\bF}{\boldsymbol{F}}$
$\newcommand{\bW}{\boldsymbol{W}}$
$\newcommand{\bQ}{\boldsymbol{Q}}$
$\newcommand{\bG}{\boldsymbol{G}}$
$\newcommand{\bS}{\boldsymbol{S}}$
$\newcommand{\bU}{\boldsymbol{U}}$
$\newcommand{\bx}{\boldsymbol{x}}$
$\newcommand{\by}{\boldsymbol{y}}$
$\newcommand{\bu}{\boldsymbol{u}}$
$\newcommand{\bv}{\boldsymbol{v}}$
$\newcommand{\bX}{\boldsymbol{X}}$
$\newcommand{\bI}{\boldsymbol{I}}$
$\newcommand{\bA}{\boldsymbol{A}}$
$\newcommand{\bB}{\boldsymbol{B}}$
$\newcommand{\bC}{\boldsymbol{C}}$
$\newcommand{\bO}{\boldsymbol{O}}$
$\newcommand{\bD}{\boldsymbol{D}}$
$\newcommand{\bJ}{\boldsymbol{J}}$
$\newcommand{\bK}{\boldsymbol{K}}$
$\newcommand{\bH}{\boldsymbol{H}}$
$\newcommand{\AR}{\text{AR}}$
$\newcommand{\Xlag}{\boldsymbol{X}_{\text{LAG}}}$
$\newcommand{\hX}{\hat{X}}$
$\newcommand{\bY}{\boldsymbol{Y}}$
$\newcommand{\bH}{\boldsymbol{H}}$
$\newcommand{\bhH}{\hat{\boldsymbol{H}}}$
$\newcommand{\bT}{\boldsymbol{T}}$
$\newcommand{\bq}{\boldsymbol{q}}$
$\newcommand{\bP}{\boldsymbol{P}}$
$\newcommand{\Gperp}{\boldsymbol{G}_{\perp}}$
$\newcommand{\bone}{\boldsymbol{1}}$
$\newcommand{\bZ}{\boldsymbol{Z}}$
$\newcommand{\bhX}{\hat{\boldsymbol{X}}}$
$\newcommand{\bhZ}{\hat{\boldsymbol{Z}}}$
$\newcommand{\bep}{\boldsymbol{\epsilon}}$
$\newcommand{\bsc}{\boldsymbol{c}}$
$\newcommand{\bhep}{\hat{\boldsymbol{\epsilon}}}$
$\newcommand{\hOmg}{\hat{\Omega}}$
$\newcommand{\btheta}{\boldsymbol{\theta}}$
$\newcommand{\blambda}{\boldsymbol{\lambda}}$
$\newcommand{\bPhi}{\boldsymbol{\Phi}}$
$\newcommand{\bpsi}{\boldsymbol{\psi}}$
$\newcommand{\baP}{\boldsymbol{\bar{P}}}$
$\newcommand{\baq}{\boldsymbol{\bar{q}}}$
$\newcommand{\bXtheta}{\boldsymbol{X}_{\theta}}$
$\newcommand{\bXthetaL}{\boldsymbol{X}_{\theta;}^L}$
$\newcommand{\bXLag}{\boldsymbol{X}_{\text{LAG}}}$
$\newcommand{\MA}{\text{MA}}$
$\newcommand{\VAR}{\text{VAR}}$
$\newcommand{\VARX}{\text{VARX}}$
$\newcommand{\VARMA}{\text{VARMA}}$
$\DeclareMathOperator{\diag}{diag}$
$\DeclareMathOperator{\Tr}{Tr}$
$\DeclareMathOperator{\Mat}{Mat}$


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
