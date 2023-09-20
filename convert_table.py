import argparse
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def prepare_table(df: pd.DataFrame, ord_mode: bool = False) -> str:
    """
    Converting pandas DataFrame table to numbered list of columns with examples of rows.

    Args:
        df (pd.DataFrame): DataFrame to convert.
        ord_mode (bool): Create numbered list with integers or with literals.

    Returns:
        data (str): Converted DataFrame to 'A: [value1, ...], ...' or '1: [value1, ...], ...'
        convert_dct (dict): Dictionary to decode convertation: {column1: A, ...} or {column1: 1, ...}
    """

    data = ""
    convert_dct = {}
    
    i = ord("A") if ord_mode else 1
    for column in df.columns:
        index = chr(i) if ord_mode else str(i)
        convert_dct[index] = column
        data += f"'{index}': {df.head(5)[column].to_list()};\n"
        i += 1
        
    return data, convert_dct


def get_convert_instructions(mapping: dict, 
                             source_df: pd.DataFrame, 
                             template_df: pd.DataFrame) -> str:
    """
    Get instructions how to map columns.

    Args:
        mapping (dict): Dict of columns from source which would map to target.
        source_df (pd.DataFrame): Source DataFrame.
        template_df (pd.DataFrame): Target DataFrame.

    Returns:
        query (str): Prompt with instructions to GPT.
    """
    
    query = """Here is instructions for working with dataframe.
    Instrunctions has examples of formats which you should convert. Don't replace data, only check for formats.
    If formats are same, skip convering instructions.
    Instrunctions:\n
    """
    for key, value in mapping.items():
        query += f"rename {key} column to {value} and convert data in it from {source_df.loc[0][key]} format to such format: {template_df.loc[0][value]}\n"
    
    query += "Provide only code without text explanation."

    return query


def get_mapping(source_table_data: str, 
                template_table_data: str, 
                convert_source: dict, 
                convert_template: dict) -> str:
    """
    Getting columns from source DataFrame which could be mapped to template DataFrame.

    Args:
        source_table_data (str): Converted source DataFrame.
        template_table_data (str): Converted template DataFrame.
        convert_source (dict): Dictionary to decode convertation of source DataFrame.
        convert_template (dict): Dictionary to decode convertation of template DataFrame.

    Returns:
        res (dict): Dict of columns from source which would map to target.
    """

    template = """
    Your task is to compare two tables and return Python dictionary of columns from first tables that can be
    mapped into columns of second table.

    Here is first table columns with few examples of content in it:

    {source_table_data}

    Here is second table columns with few examples of content in it:

    {template_table_data}
    
    By reading content examples, understand which columns from first table can be mapped into columns of second table.
    
    Also follow some rules:
    1) Find mapping for columns only by reading content examples, don't look at names. Columns should reflect the same things.
    2) Some columns in first table can be unnecessary, skip them in dictionary.
    3) If there are columns with duplicated data use first column for mapping.
    4) Return only dictionary. If there is nothing to map, answer ERROR.

    Answer:

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            'source_table_data',
            'template_table_data'])

    llm = OpenAI(
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(source_table_data=source_table_data,
                           template_table_data=template_table_data)

    if answer == 'ERROR':
        sys.exit("Unable to map source table to target.")
    else:
        try:
            mapping = eval(answer.replace('\n', ''))
            
            res = {}
            for key, value in mapping.items():
                res[convert_source[key]] = convert_template[value]
                
            return res
        except SyntaxError:
            sys.exit(f"Unable to parse model answer for mapping: {answer}")


def convert(source_path: str, template_path: str, target_path: str) -> None:
    # Reading provided csv files
    source_df = pd.read_csv(source_path).reset_index(drop=True)
    template_df = pd.read_csv(template_path).reset_index(drop=True)

    # Converting pandas format to string for model
    source_data, convert_source = prepare_table(source_df, ord_mode=True)
    template_data, convert_template = prepare_table(template_df)
    
    # Getting columns mappings
    logging.info('Checking transferability.')
    mapping = get_mapping(source_data, template_data, convert_source, convert_template)
    
    # Creating instruction for mapping
    query = get_convert_instructions(mapping, source_df, template_df)

    # Mapping source table to template format
    logging.info('Generating map instructions code.')
    llm = OpenAI(
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"))
    agent = create_pandas_dataframe_agent(llm, source_df, verbose=False)
    agent.run(query)
    
    # Saving results
    res_columns = template_df.columns
    source_df[res_columns].to_csv(target_path, index=False)
    sys.exit(f"Table was succesfully mapped.")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for mapping provided csv to template format.")
    parser.add_argument("--source", type=str, required=True, help="Path to csv which must be mapped.")
    parser.add_argument("--template", type=str, required=True, help="Path to template csv.")
    parser.add_argument("--target", type=str, required=True, help="Path to mapped target csv.")

    args = parser.parse_args()
    source_path, template_path, target_path = args.source, args.template, args.target

    if not os.path.exists(source_path) or not source_path.endswith('.csv'):
        sys.exit(
            f"Provided source path: '{source_path}' doesn't exist or not in .csv format.")

    if not os.path.exists(template_path) or not template_path.endswith('.csv'):
        sys.exit(
            f"Provided template path: '{template_path}' doesn't exist or not in .csv format.")

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("No environment variable OPENAI_API_KEY found.")

    convert(source_path, template_path, target_path)
