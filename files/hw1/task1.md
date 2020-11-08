---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# TASK 1

```python tags=["Packages"] slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Data from Premier League
Seasons including 2018/2019, 2019/2020 and 2020/2021

```python
data_2021 = pd.read_csv("C:/IE582/data/E0.csv")
data_1920 = pd.read_csv("C:/IE582/data/E1.csv")
data_1819 = pd.read_csv("C:/IE582/data/E2.csv")
match_data = pd.concat([data_2021, data_1920, data_1819], ignore_index=True)
match_data.drop(match_data.columns[56:], axis=1, inplace=True)
match_data
```

## Question 1

```python
plt.hist(match_data["FTHG"],align="left",rwidth=0.8)
plt.xlabel("Home Goals")
plt.ylabel("Number of Games")
plt.title("Home Score")
plt.show()
```

```python
plt.hist(match_data["FTAG"],color="#009E73",align="left",rwidth=0.8)
plt.xlabel("Away Goals")
plt.ylabel("Number of Games")
plt.title("Away Score")
plt.show()
```

```python
match_data["FTHG"]-match_data["FTAG"]
plt.hist(match_data["FTHG"]-match_data["FTAG"],color="black",align="left",rwidth=0.8)
plt.xlabel("Home goals – Away Goals")
plt.ylabel("Number of Games")
plt.title("Score Difference")
plt.show()
```

##  Question 2

```python
import statistics as st
from scipy.stats import poisson

home_mean = st.mean(match_data["FTHG"])
away_mean = st.mean(match_data["FTAG"])
home_mean, away_mean
```

```python
pois = poisson.pmf(np.arange(0,9), home_mean)*len(match_data)
n, bins, patches = plt.hist(match_data["FTHG"], np.arange(9),color='red',align="left",rwidth=0.8,histtype ='bar',label="Data")
plt.plot(bins, pois, color='blue', marker='o', linestyle="dashed",label="Poisson PMF")
plt.xlabel("Home Goals")
plt.ylabel("Number of Games")
plt.title("Home Score")
plt.legend(loc="upper right")
plt.show()
```

This graph shows that **Home Goals** data distribution looks like *Poisson* distribution with $ \lambda = 1.537 $ . 

```python
pois = poisson.pmf(np.arange(0,9), away_mean)*len(match_data)
n, bins, patches = plt.hist(match_data["FTAG"], np.arange(9),color='#009E73',align="left",rwidth=0.8,histtype ='bar',label="Data")
plt.plot(bins, pois, color='red', marker='o', linestyle="dashed",label="Poisson PMF")
plt.xlabel("Away Goals")
plt.ylabel("Number of Games")
plt.title("Away Score")
plt.legend(loc="upper right")
plt.show()
```

This graph shows that **Away Goals** data distribution looks like *Poisson* distribution with $ \lambda = 1.273 $ .
