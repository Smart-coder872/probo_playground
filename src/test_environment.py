import pandas as pd

# Sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Apply a lambda function to sum values across rows
result = df.apply(lambda row: row['A'] - row['B'], axis=1)
#print(result)
print(result)

