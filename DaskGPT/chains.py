import os
from typing import Optional

import dask.dataframe as dd
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.base import BaseLLM
from langchain.llms.loading import load_llm

DEFAULT_LLM = None

# Adapted Dask DataFrame creation
# For demonstration, let's assume df is a Dask DataFrame created from a CSV file
df = dd.read_csv("data.csv")

TEMPLATE = """
You are working with a Dask dataframe in Python. The name of the dataframe is `df`.
The dataframe has the following columns: {df_columns}.

You should execute code as commanded to either provide information to answer the question or to
do the transformations required.

You should not assign any variables; you should return a one-liner using Dask.

This is your objective: {query}

Go!

'''python
print(df.head())
```
```output
{df_head}
```
```python
"""

PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["query", "df_head", "df_columns"])

# Template with memory
# TODO: add result of expected code to memory; currently we only remember what code was run.
TEMPLATE_WITH_MEMORY = """
You are working with a Dask dataframe in Python. The name of the dataframe is `df`.
The dataframe has the following columns: {df_columns}.

You are interacting with a programmer. The programmer issues commands and you should translate
them into Python code and execute them.

This is the history of your interaction so far:
{chat_history}
Human: {query}

Go!

```python
df.head()
```
```output
{df_head}
```
```python
"""
PROMPT_WITH_MEMORY = PromptTemplate(
    template=TEMPLATE_WITH_MEMORY, input_variables=["chat_history", "query", "df_head", "df_columns"]
)



# Global variable to store the default language model
DEFAULT_LLM: Optional[BaseLLM] = None

def set_llm(llm: BaseLLM) -> None:
    """
    Set the default language model.
    """
    global DEFAULT_LLM
    DEFAULT_LLM = llm

def get_chain(llm: Optional[BaseLLM] = None, use_memory: bool = True) -> Chain:
    """
    Get a language model chain with optional memory support.

    Args:
        llm (Optional[BaseLLM]): The language model to use. If None, the default model is used.
        use_memory (bool, optional): Whether to use a memory buffer. Defaults to True.

    Returns:
        Chain: An instance of a language model chain.
    """
    if llm is None:
        if DEFAULT_LLM is None:
            llm_config_path = os.environ.get("LLDASK_LLM_CONFIGURATION")
            if llm_config_path is None:
                llm = OpenAI(temperature=0)
            else:
                llm = load_llm(llm_config_path)
        else:
            llm = DEFAULT_LLM

    if use_memory:
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
        chain = LLMChain(llm=llm, prompt=PROMPT_WITH_MEMORY, memory=memory)
    else:
        chain = LLMChain(llm=llm, prompt=PROMPT)

    return chain
