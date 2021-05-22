# domain-adaptation
The Python package da-properties provides useful tools for characterizing domain adaptation problems.

## Installation with pip 
 
Download a wheel (domain_adaptation_properties-X.X.X-py3-none-any.whl) from https://github.com/EUB-LE/domain-adaptation/releases/latest to your project root. Navigate to your project root and run:

```shell
pip install domain_adaptation_properties-X.X.X-py3-none-any.whl
```

## Using the package 

### Create estimated random variables from data 
Domain-adaptation-properties works by estimating random variables from empirical distributions found in machine learning training data. 

```python
import daproperties as daps 

# discrete variable 
rv_y = daps.rv_from_discrete(y) 

# continuous variable 
rv_X = daps.rv_from_continuous(X)

# all rvs from a classifier machine learning problem with continuous attributes in X and discrete labels in y: 
# rv_Xy is the common probability of X and y
rv_X, rv_y, rv_Xy = daps.rvs_from_mixed(X, y)
```

### Working with random variables 
There are three kinds of predefined random variables: discrete, continuous, and mixed. 

#### Discrete
```python
# Create discrete variable
rv_y = daps.rv_from_discrete(y) 

# calculate the probability mass function for a value 
rv_y.pmf(0) 

# calculate the probability mass function for multiple values 
rv_y.pmf([0,1,2]) 
```

#### Continuous
```python 
# Create continuous variable
rv_X = daps.rv_from_continuous(X)

# calculate the probability density function for a value or a list of values. 
rv_X.pdf([0.234, 2.345]] 
rv_X.pdf([[0.125, -0.345], [2.345, -1.234]]) 
```

#### Mixed
```python
# Create mixed variable 
rv_Xy = daps.rv_from_mixed(X,y)

# calculate common probability 
rv_Xy.pdf([[0.986, 0234]], [0])

# calculate conditional probabilities 
rv_Xy.pdf_given_y([[0.986, 0234]], [0])
rv_Xy.pmf_given_x([[0.986, 0234]], [0])
```

### Calculating divergence 
Divergences express a difference in (estimated) probability distributions. 

```python

# if two rvs have the same type (discrete, continuous, or mixed), the divergence can be calculated directly as: 
rv1.divergence(rv2, div_type="jsd") 

# measures can also be called directly 
from daproperties.measures import jsd, kld 

# Kullback-Leibler Divergence from distribution 
kld(pd_x, pd_y)
```
