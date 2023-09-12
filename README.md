# Model is still in development

# :robot: DaskGPT - Your Interactive Data Analysis Assistant

This repository showcases the implementation of an interactive data analysis assistant using Language Model Chains (LLMs) and Dask DataFrames.

DaskGPT is designed to assist you in interactive data exploration and analysis, making use of language models to provide code snippets based on your queries. 


## Quickstart

```python
import os
import dask.dataframe as dd
from DaskGPT.chains import get_chain

os.environ['LLDASK_API_KEY'] = 'your-api-key-here' # Fill in your API key

# Load your Dask DataFrame (replace "data.csv" with your dataset)
df = dd.read_csv("data.csv")

# Initialize the DaskGPT assistant
dask_gpt = get_chain()

# Ask a question or request a data analysis task
query = "Show me the top 5 rows of the DataFrame."
result = dask_gpt.query(query)

# Display the result
print(result)
```

DaskGPT is your companion for interactive data analysis. Simply provide a query, and DaskGPT will generate the code to fulfill your request.

## Understanding DaskGPT

DaskGPT offers interactive data analysis capabilities using Dask DataFrames. It understands your data and task and responds accordingly, making your data analysis/ programming tasks more efficient.

## Installation

***Ensure you have Python 3.7+ installed and run:
#in development
```bash
pip install -r requirements.txt
pip install DaskGPT
```
