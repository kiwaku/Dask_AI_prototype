import ast
import os
import dask.dataframe as dd
from IPython.display import clear_output
from langchain.chains.base import Chain
from langchain.input import print_text
from langchain.llms.base import BaseLLM
from DaskGPT.chains import get_chain

from typing import Any, Optional

@dd.extensions.register_dataframe_accessor("llm")
class LLMAccessor:
    def __init__(self, dask_df: dd.DataFrame):
        self.df = dask_df
        self.use_memory = bool(os.environ.get("LLDASK_USE_MEMORY", True))
        self.chain = get_chain(use_memory=self.use_memory)

    def set_chain(self, chain: Chain) -> None:
        """
        Set the language model chain.
        """
        self.chain = chain

    def reset_chain(self, llm: Optional[BaseLLM] = None, use_memory: bool = True) -> None:
        """
        Reset the language model chain.
        """
        self.chain = get_chain(llm=llm, use_memory=use_memory)

    def query(self, query: str, yolo: bool = False) -> Any:
        """
        Execute a query using the language model chain.

        Args:
            query (str): The query to be executed.
            yolo (bool, optional): If True, execute without confirmation. Defaults to False.
        """
        df = self.df
        df_columns = df.columns.tolist()
        df_head = df.head().compute()  # Compute the head for display
        inputs = {"query": query, "df_head": df_head, "df_columns": df_columns, "stop": "```"}
        llm_response = self.chain.run(**inputs)
        eval_expression = False
        if not yolo:
            print("Suggested code:")
            print(llm_response)
            print("Run this code? y/n")
            user_input = input()
            if user_input == "y":
                clear_output(wait=True)
                print_text(llm_response, color="green")
                eval_expression = True
        else:
            eval_expression = True

        if eval_expression:
            tree = ast.parse(llm_response)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module))
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)
            try:
                return eval(module_end_str)
            except Exception:
                exec(module_end_str)
