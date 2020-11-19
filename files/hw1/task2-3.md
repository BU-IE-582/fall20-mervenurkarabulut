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

# Task 2

```python
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

```python
probs = 1 / match_data[["B365H","B365D","B365A","BWH","BWD","BWA","PSH","PSD","PSA","WHH","WHD","WHA"]]
match_data[["B365H_prob","B365D_prob","B365A_prob","BWH_prob","BWD_prob","BWA_prob","PSH_prob","PSD_prob","PSA_prob",
            "WHH_prob","WHD_prob","WHA_prob"]] = probs
match_data[["B365H_prob","B365D_prob","B365A_prob","BWH_prob","BWD_prob","BWA_prob","PSH_prob","PSD_prob","PSA_prob",
            "WHH_prob","WHD_prob","WHA_prob"]]
```

These are the **probabilities** of "home win", "draw" and "away win" given by four bookmakers respectively: **Bet365, BetAndWin, Pinnacle and William Hill**.

```python
match_data["Bet365_sum"] = match_data.apply(lambda x: sum(x[["B365H_prob","B365D_prob","B365A_prob"]]),axis=1)
match_data["BetAndWin_sum"] = match_data.apply(lambda x: sum(x[["BWH_prob","BWD_prob","BWA_prob"]]),axis=1)
match_data["Pinnacle_sum"] = match_data.apply(lambda x: sum(x[["PSH_prob","PSD_prob","PSA_prob"]]),axis=1)
match_data["ABookmaker_sum"] = match_data.apply(lambda x: sum(x[["WHH_prob","WHD_prob","WHA_prob"]]),axis=1)

match_data[["B365H_norm_prob","B365D_norm_prob","B365A_norm_prob"]] = match_data.apply(
                                                lambda x: x[["B365H_prob","B365D_prob","B365A_prob"]]/x["Bet365_sum"], axis=1)
match_data[["BWH_norm_prob","BWD_norm_prob","BWA_norm_prob"]] = match_data.apply(
                                                lambda x: x[["BWH_prob","BWD_prob","BWA_prob"]]/x["BetAndWin_sum"], axis=1)
match_data[["PSH_norm_prob","PSD_norm_prob","PSA_norm_prob"]] = match_data.apply(
                                                lambda x: x[["PSH_prob","PSD_prob","PSA_prob"]]/x["Pinnacle_sum"], axis=1)
match_data[["WHH_norm_prob","WHD_norm_prob","WHA_norm_prob"]] = match_data.apply(
                                                lambda x: x[["WHH_prob","WHD_prob","WHA_prob"]]/x["ABookmaker_sum"], axis=1)

match_data[["B365H_norm_prob","B365D_norm_prob","B365A_norm_prob","BWH_norm_prob","BWD_norm_prob","BWA_norm_prob",
           "PSH_norm_prob","PSD_norm_prob","PSA_norm_prob","WHH_norm_prob","WHD_norm_prob","WHA_norm_prob"]]
```

These are the **normalized probabilities** of "home win", "draw" and "away win" given by four bookmakers respectively: **Bet365, BetAndWin, Pinnacle and William Hill**.

```python
match_data["draw"] = match_data["FTR"].apply(lambda x: 1 if x=="D" else 0)
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)
    
match_data["B365_diff"] = match_data["B365H_norm_prob"]-match_data["B365A_norm_prob"]
match_data["range"] = pd.cut(match_data["B365_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()

plt.scatter(match_data["B365_diff"], match_data["B365D_norm_prob"], marker='o',c="tab:orange",edgecolors="black",
            label="Bet365 probabilities")
plt.plot(bin_ranges,actual["draw"],color="lime",label="Actual outcome")
plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("Bet365")
plt.legend(loc="lower center")
plt.show()
```

```python
match_data["BW_diff"] = match_data["BWH_norm_prob"]-match_data["BWA_norm_prob"]
match_data["range"] = pd.cut(match_data["BW_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()

plt.scatter(match_data["BW_diff"], match_data["BWD_norm_prob"], marker='o',c="g",edgecolors="black",
            label="BetAndWin probabilities")
plt.plot(bin_ranges,actual["draw"],color="fuchsia",label="Actual outcome")
plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("BetAndWin")
plt.legend(loc="lower center")
plt.show()
```

```python
match_data["PS_diff"] = match_data["PSH_norm_prob"]-match_data["PSA_norm_prob"]
match_data["range"] = pd.cut(match_data["PS_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()

plt.scatter(match_data["PS_diff"], match_data["PSD_norm_prob"], marker='o',edgecolors="black",label="Pinnacle probabilities")
plt.plot(bin_ranges,actual["draw"],color="red",label="Actual outcome")
plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("Pinnacle")
plt.legend(loc="lower center")
plt.show()
```

```python
match_data["WH_diff"] = match_data["WHH_norm_prob"]-match_data["WHA_norm_prob"]
match_data["range"] = pd.cut(match_data["WH_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()

plt.scatter(match_data["WH_diff"], match_data["WHD_norm_prob"], marker='o',c="purple",edgecolors="black",label="William Hill probabilities")
plt.plot(bin_ranges,actual["draw"],color="aqua",label="Actual outcome")
plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("William Hill")
plt.legend(loc="lower center")
plt.show()
```

It can be seen in the graphs above, where `0 < P(home)-P(away) < 0.5` many of the bins for these games are above the given odd probabilities. So, there is a positive bias in these odds.


# Task 3

```python
match_data["TotalRedCards"] = match_data["HR"] + match_data["AR"]
without_redCard = match_data[match_data["TotalRedCards"] == 0].copy()

without_redCard["range"] = pd.cut(without_redCard["B365_diff"], bins, right=True, labels=False) + 1
actual = without_redCard.groupby(["range"]).sum()/without_redCard.groupby(["range"]).count()

plt.scatter(without_redCard["B365_diff"], without_redCard["B365D_norm_prob"], marker='o',c="tab:orange",edgecolors="black",
            label="Bet365 probabilities")
plt.plot(bin_ranges,actual["draw"],color="green",label="Actual outcome with no red cards")

match_data["range"] = pd.cut(match_data["B365_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()
plt.plot(bin_ranges,actual["draw"],color="lime",label="Actual outcome")

plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("Bet365")
plt.legend(loc="lower center")
plt.show()
```

```python
without_redCard["range"] = pd.cut(without_redCard["BW_diff"], bins, right=True, labels=False) + 1
actual = without_redCard.groupby(["range"]).sum()/without_redCard.groupby(["range"]).count()

plt.scatter(without_redCard["BW_diff"], without_redCard["BWD_norm_prob"], marker='o',c="g",edgecolors="black",
            label="BetAndWin probabilities")
plt.plot(bin_ranges,actual["draw"],color="blue",label="Actual outcome with no red cards")

match_data["range"] = pd.cut(match_data["BW_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()
plt.plot(bin_ranges,actual["draw"],color="fuchsia",label="Actual outcome")

plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("BetAndWin")
plt.legend(loc="lower center")
plt.show()
```

```python
without_redCard["range"] = pd.cut(without_redCard["PS_diff"], bins, right=True, labels=False) + 1
actual = without_redCard.groupby(["range"]).sum()/without_redCard.groupby(["range"]).count()
plt.scatter(without_redCard["PS_diff"], without_redCard["PSD_norm_prob"], marker='o',edgecolors="black",
            label="Pinnacle probabilities")
plt.plot(bin_ranges,actual["draw"],color="gold",label="Actual outcome with no red cards")

match_data["range"] = pd.cut(match_data["PS_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()
plt.plot(bin_ranges,actual["draw"],color="red",label="Actual Outcome")

plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("Pinnacle")
plt.legend(loc="lower center")
plt.show()
```

```python
without_redCard["range"] = pd.cut(without_redCard["WH_diff"], bins, right=True, labels=False) + 1
actual = without_redCard.groupby(["range"]).sum()/without_redCard.groupby(["range"]).count()

plt.scatter(without_redCard["WH_diff"], without_redCard["WHD_norm_prob"], marker='o',c="purple",edgecolors="black",
            label="William Hill probabilities")
plt.plot(bin_ranges,actual["draw"],color="tab:orange",label="Actual outcome with no red cards")

match_data["range"] = pd.cut(match_data["WH_diff"], bins, right=True, labels=False) + 1
actual = match_data.groupby(["range"]).sum()/match_data.groupby(["range"]).count()
plt.plot(bin_ranges,actual["draw"],color="aqua",label="Actual Outcome")

plt.xlabel("P(Home)-P(Away)")
plt.ylabel("P(Draw)")
plt.title("William Hill")
plt.legend(loc="lower center")
plt.show()
```

After eliminating games with red cards, bins where `P(home)-P(away) > 0` are slightly shifted above. So, there is a slight improvement in given odds for these games.
