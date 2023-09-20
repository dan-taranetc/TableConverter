# TableConverter

## Usage

1. Install dependencies from `requirements.txt`;
2. Create environment variable `OPENAI_API_KEY` with openai api key;
3. Run script:
```bash
python3 convert_table.py --source='source_csv' --template='template_csv' --target='target_csv'
```

## Solution

This solution based on OpenAI **text-davinci-003** connected with LangChain. 

First of all, model understands which columns from source table can be mapped into columns of template table. 
The former table contains column names that are converted to literals (e.g. *Date_of_Policy* to *A*, *FullName* to *B*, etc.), 
while the latter contains names  that are converted to integers (e.g. *Date* to *1*, *EmployeeName* to *2*, etc.). 
It was observed that in this way the model is not bound to specific names and has a better understanding of which columns are linked to which ones by the same data.


Then source table and mapping dictionary from first model are provided into the second model, which is a LangChain Pandas DataFrame agent, that can work with DataFrames. 
It renames columns and converts data to necessary format.

