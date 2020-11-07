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
plt.hist(match_data["FTHG"])
plt.xlabel("Home Goals")
plt.ylabel("Number of Games")
plt.title("Home Score")
plt.show()
```

```python
plt.hist(match_data["FTAG"],color="#009E73")
plt.xlabel("Away Goals")
plt.ylabel("Number of Games")
plt.title("Away Score")
plt.show()
```

```python
match_data["FTHG"]-match_data["FTAG"]
plt.hist(match_data["FTHG"]-match_data["FTAG"],color="black")
plt.xlabel("Home goals – Away Goals")
plt.ylabel("Number of Games")
plt.title("Score Difference")
plt.show()
```

##  Question 2

```python
import statistics as st

home_mean = st.mean(match_data["FTHG"])
away_mean = st.mean(match_data["FTAG"])
home_mean, away_mean
```

```python
plt.hist(match_data["FTHG"],density=True,label="Data")
plt.xlabel("Home Goals")
plt.ylabel("Probability")
plt.title("Home Score")
mn, mx = plt.xlim()
probs = []
for score in np.arange(mn,mx+1):
    probs.append(poisson.pmf(score,home_mean))
plt.plot(probs, label="Poisson PMF")
plt.legend(loc="upper right")
plt.show()
```

This graph shows that **Home Goals** data distribution looks like *Poisson* distribution with $ \lambda = 1.537 $ . 

```python
plt.hist(match_data["FTAG"],density=True,label="Data",color="#009E73")
plt.xlabel("Away Goals")
plt.ylabel("Probability")
plt.title("Away Score")
mn, mx = plt.xlim()
probs = []
for score in np.arange(mn,mx+1):
    probs.append(poisson.pmf(score,away_mean))
plt.plot(probs, label="Poisson PMF")
plt.legend(loc="upper right")
plt.show()
```

This graph shows that **Away Goals** data distribution looks like *Poisson* distribution with $ \lambda = 1.273 $ .
