## Introduction

This is the second of three exercises that will give you a solid foundation for doing BLP-style estimation. We'll continue with the same running example: what if we halved an important product's price? Our goal today is to relax some of the unrealistic substitution patterns implied by the pure logit model by incorporating preference heterogeneity. To do so, we will use cross-market variation in our product and some new demographic data to estimate parameters that govern preference heterogeneity.

## Setup

We'll be continuing where we left off after the [first exercise](https://github.com/Mixtape-Sessions/Demand-Estimation/blob/main/Exercises/Exercise-1/README.md). You should just keep adding to your code, using [its solutions](https://github.com/Mixtape-Sessions/Demand-Estimation/blob/main/Exercises/Exercise-1/Solutions.ipynb) if you had trouble with some parts.

## Data

Today, you'll incorporate [`demographics.csv`](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/demographics.csv) into estimation, which again is a simplified version of [Nevo's (2000)](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Readings/5-Nevo-2000.pdf) demographic data with less information and fewer derived columns. The data were originally draws from the Current Population Survey.

In your own work, when incorporating demographic data into estimation, you will want to sample from the whole Current Population Survey (or whatever survey/census data you are using), not just from a subset of it. The small size of today's demographic data helps with distributing the data, but in practice you should ideally be sampling from a much larger dataset of demographic information. In your own work you will also want to incorporate more demographic variables than the one included in this dataset. Like the product data, in these exercises we only consider a few columns to keep the exercises a manageable length.

The demographic dataset contains information about 20 individuals drawn from the Current Population Survey for each of the 94 markets in the product data. Each row is a different individual. The columns in the data are as follows.

Column             | Data Type | Description
------------------ | --------- | -----------
`market`           | String    | The city-quarter pair that defines markets $t$ used in these exercises. The data were motivated by real cereal purchase data across 47 US cities in the first 2 quarters of 1988.
`quarterly_income` | Float     | The quarterly income of the individual in dollars.

In today and tomorrow's exercises, we will use these demographic data to introduce income-specific preference heterogeneity into our BLP model of demand for cereal and see how our counterfactual changes.

## Questions

### 1. Describe cross-market variation

To get a sense for what types of preference heterogeneity we can feasibly estimate with variation in our product and demographic data, recall the linear regression intuition about identification from the lecture. To add parameters in $\Sigma$ we want cross-market choice set variation. To add parameters in $\Pi$ we want cross-market demographic variation.

You should already have `products.csv` from the last exercise. You can download `demographics.csv` from [this link](https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/demographics.csv). Quarterly income has a long right tail that makes summarizing it difficult, so you should create a new column `log_income` equal to the log of `quarterly_income`.

Across both datasets, describe the amount of cross-market variation in the number of products, `mushy`, `prices`, and `log_income`. You can use [`.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) and [`.agg`](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) to compute market-level statistics for these variables (e.g., counts, means, and standard deviations), and you can then use [`.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) to compute summary statistics for these market-level statistics.

Which variables have *any* cross-market variation? In other words, referring back to the linear regression intuition about identification, which preference heterogeneity parameters do we have any hope of credibly estimating? Recall that we have both product and market fixed effects, which will be collinear with some regressors on potential parameters in $\Sigma$ and $\Pi$ in the approximate linear regression.

### 2. Estimate a parameter on mushy $\times$ log income

To estimate a parameter in $\Pi$ on the interaction between `mushy` and `log_income`, we first need to define a datset of consumer types that PyBLP can use for preference heterogeneity. From each market $t$, take $I_t = 1,000$ draws with replacement from the demographic data, and call the resulting dataframe `agent_data`. Each of its rows will be a consumer type $i \in I_t$. You can do this with [`.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) and [`.sample`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sample.html). You can use `as_index=False` when grouping to keep your `market` column. Remember to set your seed with the `random_state` argument when sampling from each market.

Like with product data, to get PyBLP to recognize the `market` column as markets $t$, you'll need to rename it to `market_ids`. You'll also need to specify consumer type shares / sampling weights $w_{it}$. Since we took each draw from the demographic data with equal probability, these should be uniform weights $w_{it} = 1 / 1,000$. Make a new column `weights` equal to `1 / 1000`.

Finally, later we'll be adding some dimensions of unobserved preference heterogeneity in $\nu_{it}$. PyBLP recognizes these with the column names `nodes0`, `nodes1`, `nodes2`, etc. Make three columns, `nodes0`, `nodes1`, and `nodes2`, each with a *different* set of draws from the standard normal distribution. You can do this with [`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html), remembering to set your `seed`, and [`.normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal), using `size=len(agent_data)`. Print some rows from your `agent_data` to make sure it looks like you'd expect.

To identify our new parameter, we need a new instrument. We'll use the recommendation from the lecture to use mean income $m_t^y$ interacted with `mushy` in $x_{jt}$. To merge in the mean market-level income from the first question into the product data, you can use [`.merge`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). Recall that you already have one column `demand_instruments0` equal to your price instrument. To add a second instrument, create a new column `demand_instruments1` equal to the interaction between your merged-in market-level mean income and `mushy`.

When initializing your new [`pyblp.Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html), you'll need two new [`pyblp.Formulation`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Formulation.html) instances to model the interaction between `mushy` and `log_income`. First, replace your old formulation with a tuple

```python
product_formulations = (pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'), pyblp.Formulation('0 + mushy'))
```

This defines, in PyBLP lingo, the formulations for the `X1` and `X2` matrices. The `X1` matrix is the one with `beta` coefficients. The `X2` matrix is interacted with consumer type-specific variables like your demographics $y_{it}$. There `0` values in the formulations guarantee that they won't have constant terms (by default a constant is added, unless there are absorbed fixed effects). You'll also need a new formulation for consumer demographics.

```python
agent_formulation = pyblp.Formulation('0 + log_income')
```

With these in hand, we can define the new problem. See the [`pyblp.Problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html) documentation for the ordering and names of arguments.

```python
pyblp.Problem(product_formulations, product_data, agent_formulation, agent_data)
```

Now that we want to set up a nonlinear GMM problem, we need to configure some nonlinear optimization arguments in [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html). First, we should configure our optimizer. We'll use the lecture's recommendation to use SciPy's trust region algorithm [`trust-constr`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html). You can do this by setting `optimization=pyblp.Optimization('trust-constr')`. In [`pyblp.Optimization`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Optimization.html), you can configure `method_options` with a [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) mapping [`trust-constr`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html) configuration options to their values. In this case the default will work, but you should try changing up some configuration options and see how it affects optimization.

Second, we need to specify which parameters in $\Sigma$ and $\Pi$ to optimize over, and what their initial values will be. Since we're starting without any parameters in $\Sigma$, we'll set `sigma=0`. Zeros in PyBLP mean that PyBLP should have the parameter *always* be set to zero, i.e. not there. There's only one new parameter in $\Pi$ (there's only one column in your `X2` formulation, `mushy`, and only one demographic in your agent formulation, `log_income`), and we'll just set it to an arbitrary nonzero value for now (indicating that it should be optimized over), say `pi=1`.

Before and after you solve the problem, you can set `pyblp.options.verbose = True` and `pyblp.options.verbose = False`. Make sure that the gradient norm (the optimization problem's first-order conditions) is getting iteratively closer to zero. We call this "marching down the gradient" and not seeing it near the end of optimization indicates a problem. Since we have exactly as many parameters as moments/instruments, we're *just identified* and should also have an approximately zero objective at the optimum, so make sure you see that as well. At the optimum, in addition to a near-zero objective and gradient norm, the Hessian (in this case just a single value) should be positive, indicating that the optimization's second-order conditions are satisfied. The other columns outputted in the optimization progress have information about the inner loop, e.g. how many iterations it takes to solve it and how long this takes.

Once you're satisfied that optimization seems to have worked well, interpret the sign of your estimated parameter in $\Pi$. Can you use $\hat{\alpha}$ to interpret it in dollar terms, i.e. how much more a person with 1% higher income is willing to pay for mushyness?

### 3. Make sure you get the same estimate with random starting values

We can be fairly confident with our objective/gradient/Hessian checks that we're at the global optimum with just one parameter in a just identified model. But with more complicated optimization problems in more realistic problems, we really want to try different starting values to make sure we always get the same estimates. Let's try doing that here.

First, configure some bounds for our new $\Pi$ parameter, say `pi_bounds = (-10, +10)`. Then in a [for loop](https://wiki.python.org/moin/ForLoop) over different random number generator seeds, randomly draw an initial starting value from within these bounds (you can use [`.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html)) and re-optimize. You may want to actually impose your bounds during optimization using the `pi_bounds` argument to [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html).

Do you get the same estimate for each random starting value? If not, there may be an issue with optimization or your problem configuration. If you have a more complicated model with many parameters and many instruments, you may often get a global minimum, and sometimes get a local minimum. Optimizers aren't perfect, and sometimes terminate prematurely, even with tight termination conditions. You should select the global one for your final estimates.

### 4. Evaluate changes to the price cut counterfactual

Using the new estimates, re-run the same price cut counterfactual from last exercise. Re-compute percent changes and compare with those from the pure logit model. Are there any (small) differences, particularly for substitution from other products? Explain how the introduction of the new parameter induced these changes. Do cannibalization estimates seem more reasonable than before?

### 5. Estimate parameters on price $\times$ log income and unobserved preferences

Like for mushy, our cross-market income variation allows us to estimate a second parameter in $\Pi$ on `prices` $\times$ `log_income`. Unlike mushy, which doesn't vary across markets, we actually have cross-market variation in prices, which will allow us to in theory credibly estimate a parameter in $\Sigma$ on `prices`.

To add these two new parameters, we'll need two new instruments. Since we can't have endogenous prices in our instruments, we'll first set a new column `predicted_prices`  equal to the fitted values from the price IV first stage regression that we ran yesterday. If you used `statsmodels`, you can just get these from [`.fittedvalues`](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.fittedvalues.html) of your regression results object. Verify that `prices` and `predicted_prices` are strongly correlated, for example with [`.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html).

To target the new parameter in $\Pi$, we'll follow the lecture's recommendation and add a new `demand_instruments2` equal to the interaction between the log income mean and `predicted_prices`. To target the new parameter in $\Sigma$, we'll also follow the lecture's recommendation and add a new `demand_instruments3` equal to the sum of squared distances between `predicted_prices` and all other `predicted_prices` in the same market. You could construct this market-by-market by using [`.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) and [`.transform`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html) with a custom [function](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) that accepts a market's `predict_prices` as an argument, computes a matrix of all pairwise differences between these values (e.g., with `x[:, None] - x[None, :]`), squares them, and [sums](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) them across columns.

When initializing the problem, we'll need to add a new `prices` term in the `X2` formulation: `0 + mushy + prices`. Otherwise, with the updated product data, initializing the problem is the same. When solving the problem, however, the extra column in the `X2` formulation means that we need an extra row in our configuration for `sigma` and `pi`. First, `sigma` should be a $2 \times 2$ matrix corresponding to the two columns in `X2`. All elements will be zero (indicating that the corresponding elements in $\Sigma$ will be fixed to zero), except for the bottom-right value, which we'll set to some arbitrary non-zero starting values, say `1`. Similarly, `pi` should be a $2 \times 1$ matrix corresponding to the two columns in `X2` and the one column in the agent formulation. We'll set the top element equal to some starting value that's close to our last estimate, say `0.2`, and the bottom element equal to something arbitrary for the new parameter, say `1` again. Your `sigma` and `pi` arguments should look like

```python
sigma=[
    [0, 0],
    [0, 1],
], 
pi=[
    [0.2],
    [1],
]
```

When you solve the problem, there are two more parameters, so it will take a bit longer. As the optimizer nears convergence, we should again see "marching down the gradient" and, since we're still just identified, an objective approaching zero. At the optimum, verify that the gradient norm is again near zero and that the Hessian's eigenvalues are all positive. Usually, we would again draw multiple starting values to be sure everything's working fine, but from here on out, to save time we'll just use one set of starting values.

Note that your new random coefficient on price is $\alpha_{it} = \alpha + \sigma_p \nu_{1it} + \pi_p y_{it}$ where $\nu_{1it}$ is `nodes0` in your `agent_data` and $y_{it}$ is `log_income`. Compute the average log income $\bar{y}$ in your demographic data to verify that your estimate of the *average* price coefficient $\alpha + \sigma_p \times 0 + \pi \times \bar{y}$ is close to your old $\hat{\alpha}$. Using this average price sensitivity, interpret the two new parameters. Does price sensitivity vary a lot or a little with income? Compare to heterogeneity induced by income, is there a lot or a little unobserved heterogeneity in price sensitivity?

### 6. Evaluate changes to the price counterfactual

Re-run the price counterfactual and discuss new differences. Do substitution and cannibalization seem more reasonable than before? Is there anything that we were unable to estimate with cross-market product-level variation that you think could have helped further improve substitution patterns?

## Supplemental Questions

These questions will not be directly covered in lecture, but will be useful to think about when doing BLP-style estimation in your own work.

### 1. Use different numbers of Monte Carlo draws

Above, you took $I_t = 1,000$ draws per market when constructing your agent data. Particularly for simple models like this, this is usually a sufficient number of draws. In practice, however, when there are many more markets and dimensions of heterogeneity, you may want to start with a smaller number like $I_t = 100$ to get started, and then increase this number until your estimates stop changing and optimization stops having any issues. Try re-estimating the model with 10, 100, 500, and 2,000 draws, and see how your estimates change, if at all.

### 2. Use scrambled Halton sequences

Using a random number generator like [`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html) is perhaps the simplest approach to approximate an integral over a distribution like standard normals. However, using quasi-Monte Carlo sequences can do better with fewer draws. Instead of using $N(0, 1)$ draws from a random number generator, try using scrambled Halton draws.

You can do so with [`scipy.stats.qmc.Halton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html) or [`pyblp.build_integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.build_integration.html#pyblp.build_integration) with `specification='halton'` in [`pyblp.Integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html). Again, estimate the model with 10, 100, 500, 1,000, and 2,000 quasi-Halton numbers per market, and see how the stability of your estimates compares to when you use simple Monte Carlo draws.

### 3. Use quadrature

A particularly computationally-efficient approach to approximating a Gaussian distribution is with [quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature). If you're working with a model where consumer preference heterogeneity is not particularly complicated, you may want to try to replace Monte Carlo or quasi-Monte Carlo draws with many fewer nodes/weights that very well approximate the distribution.

You can construct quadrature nodes and weights with  [`pyblp.build_integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.build_integration.html#pyblp.build_integration) with `specification='product'` in [`pyblp.Integration`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html). The `'product'` specification means that PyBLP will take the cross-product of the quadrature nodes and weights for each univarate $N(0, 1)$. If you have, say, `size=7` nodes/weights per market for each $N(0, 1)$ but 5 dimensions of heterogeneity (i.e., `nodes0`, `nodes1`, `nodes2`, `nodes3`, and `nodes4`), this will give $7^5 = 16,807$ consumer types per market. This is known as the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), which Monte Carlo integration does not suffer from as much. With this many dimensions of heterogeneity, you could either switch to back to Monte Carlo integration or try using [sparse grid](https://en.wikipedia.org/wiki/Sparse_grid) integration, which more carefully chooses quadrature nodes/weights in higher dimensions. You can do this with `specification='grid'`.

In our setting, we are only using one set of $N(0, 1)$ draws for unobserved preference heterogenity for price. We can use `specification='product'`, `size=7`, and `dimensions=2` to construct two $N(0, 1)$ draws, the second of which we'll want to convert into income draws. We can do this by estimating a lognormal distribution for income in each market (you'll need to compute market-specific means and standard deviations of log income), and transforming the second column of $N(0, 1)$ draws into log income draws. Try doing this and see if your estimates (and compute time) differs much from before. If they do, this indicates that a lognormal distribution for income may not be a great parametric assumption, and a Monte Carlo approach may have made more sense than quadrature.

### 4. Incorporate supply-side restrictions into estimation

In the supplemental exercises of the first exercise, we used a canonical assumption about how firms set prices to impute marginal costs of producing each cereal from pricing optimality conditions. If we are further willing to model the functional form of marginal costs, for example as $c_{jt} = x_{jt}'\gamma + \omega_{jt}$, we can stack our current moment conditions $E[\xi_{jt} z_{jt}^D]$ with additional supply-side moment conditions $E[\omega_{jt} z_{jt}^S]$. These will allow us to identify $\gamma$, and can also help to add precision to our demand-side estimates. Intuitively, more information about marginal costs gives us more information about firms' markups, which depend on price elasticities and hence the different parameters that govern the demand side of the model.

Like the the first exercise, you will need a column of `firm_ids` to tell PyBLP which firms produce what cereals. To impose the assumption that $c_{jt} = x_{jt}'\gamma + \omega_{jt}$ for some characteristics $x_{jt}$, you'll need to specify a third formulation in your `product_formulations` tuple. For simplicity, try assuming that the only observed characterstics that affect marginal costs are a constant and the `mushy` dummy.

```python
product_formulations = (
    pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'), 
    pyblp.Formulation('0 + mushy'), 
    pyblp.Formulation('1 + mushy'),
)
```

By default, PyBLP knows that a constant and `mushy` are exogenous variables, so it will add them to the set of supply-side instruments $z_{jt}^S$. If you wanted to model non-constant returns to scale, for example by including `I(market_size * shares)` to have a coefficient on quantities $q_{jt} = M_t \cdot s_{jt}$, PyBLP would recognize that this term included endogenous market shares and not include it in $z_{jt}^S$. Instead, you would have to specify a `supply_instruments0` column in your product data to give a valid demand-shifter for market shares. Ideally, you would want something that affects demand but that is uncorrelated with the unobserved portion of marginal costs $\omega_{jt}$.

Try estimating the model with supply-side restrictions. How do your estimates change? Try adding product or market fixed effects to the supply-side configuration.

### 5. Approximate the optimal instruments and weights

After estimating your model, let PyBLP approximate the optimal instruments with [`.compute_optimal_instruments`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.compute_optimal_instruments.html). You can then use the method [`.to_problem`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.OptimalInstrumentResults.to_problem.html) to automatically update your problem to include the optimal instruments. 

If you include a supply side, you will have more optimal instruments than parameters, so your mode. will be over identified. This means that your objective will not necessarily be close to zero at the optimum, and the GMM weighting matrix $W$ will matter for statistical efficiency. When calling [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html) on your updated problem, set `sigma` and `pi` equal to your first-stage estimates [`.sigma`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.sigma) and [`.pi`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.pi), and `optimization=pyblp.Optimization('return')` to get an estimate [`.updated_W`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.ProblemResults.html#pyblp.ProblemResults.updated_W) of the optimal weighting matrix for your new optimal instruments. Pass this to the `W` argument of [`.solve`](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html) in your updated problem to use both the optimal instruments and the optimal weighting matrix for a second GMM step.
