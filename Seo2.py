from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from apikey import APIKEY2
import os
import json
import re

os.environ["GOOGLE_API_KEY"] = APIKEY2

def indexAndStats(url):
    try:
        main_template = PromptTemplate(
            input_variables = ['url'],
            template = 'Check wether {url} is indexed by google, Make sure to answer in json with key "is_indexed_by_google" with a True or False'
        )

        model = GooglePalm(temperature=0.1)

        title_chain = LLMChain(llm = model, prompt = main_template, verbose = False, output_key = 'details')

        ans = title_chain.run(url)

        data = json.loads(ans)

        # Extract the features
        google_index = data.get("is_indexed_by_google")
        return google_index
    except Exception as e:
        google_index = False
        return google_index
