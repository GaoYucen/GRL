#%%
import random

result = [(random.random() ** 2) * 0.15 for _ in range(10000)]

print(result)