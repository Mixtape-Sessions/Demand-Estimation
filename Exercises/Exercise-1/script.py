"""Script template."""




import pyblp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# You can read the product data directly from its URL.
product_data = pd.read_csv('https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/products.csv')


product_data.describe()

#%%
product_data['market_size'] = product_data['city_population'] * 90

product_data['market_share'] = product_data['servings_sold'] / product_data['market_size']
product_data['outside_share'] = 1 - product_data['market_share'].groupby(product_data['market']).transform('sum')

print(product_data.describe())

#%%


product_data['log_delta'] = np.log(product_data['market_share'] / product_data['outside_share'])


model = smf.ols("log_delta ~ mushy + price_per_serving", data=product_data).fit(cov_type="HC0")

print(model.summary())


wtp_mushy= -model.params['mushy'] / model.params['price_per_serving']

print(f"The willingness to pay for mushy peas is ${wtp_mushy:.2f} per serving.")

#%%
product_data.rename(columns={'market': 'market_ids', 'product': 'product_ids', 'market_share': 'shares', 'price_per_serving': 'prices'}, inplace=True)
product_data['demand_instruments0'] = product_data['prices']

ols_problem = pyblp.Problem(pyblp.Formulation('1 + mushy + prices'), product_data)
ols_results = ols_problem.solve(method='1s')
print(ols_results)

olsFE_problem = pyblp.Problem(pyblp.Formulation('prices', absorb='C(market_ids)+C(product_ids)'),product_data)
olsFE_results = olsFE_problem.solve(method='1s')
print(ols_results)
print(olsFE_results)
#%%

first_stage_model = smf.ols("prices ~ price_instrument + C(market_ids) + C(product_ids)", data=product_data).fit(cov_type="HC0")
first_stage_model.summary()

product_data['demand_instruments0'] = product_data['price_instrument']

IVFE_problem = pyblp.Problem(pyblp.Formulation('0 + prices', absorb='C(market_ids)+C(product_ids)'),product_data)
IVFE_results = IVFE_problem.solve(method='1s')
print(IVFE_results)

#%%
counterfactual_data = product_data[product_data['market_ids']=='C01Q2'].copy()
counterfactual_data['new_prices'] = counterfactual_data['prices'] 
mask = counterfactual_data['product_ids'] == 'F1B04'
counterfactual_data.loc[mask, 'new_prices'] = counterfactual_data.loc[mask, 'new_prices'] * 0.5

new_shares=IVFE_results.compute_shares(prices=counterfactual_data['new_prices'], market_id='C01Q2')
counterfactual_data['new_shares'] = new_shares
counterfactual_data['change_in_shares'] = counterfactual_data['new_shares'] - counterfactual_data['shares']

elast=IVFE_results.compute_elasticities( market_id='C01Q2')
