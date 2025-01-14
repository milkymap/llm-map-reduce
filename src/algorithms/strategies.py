import json 

from pydantic import BaseModel, Field 
from openai import OpenAI
from openai.types.chat import ChatCompletion

from typing import List, Dict, Tuple, Type, Optional 

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from src.algorithms.prompts import map_system_prompt, reduce_system_prompt
from src.log import logger 

def llm_map(page:str, query:str, model:str, llm:OpenAI) -> str:
    logger.info('map phase')
    out_map:ChatCompletion = llm.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": map_system_prompt
            }, 
            {
                "role": "user", 
                "content": f"""query: {query}
                Page content: {page}
                Extract the relevant information from this page that helps address the query.
                """
            }
        ],
        max_tokens=1024
    )
    stringyfied_data = out_map.choices[0].message.content
    return stringyfied_data

def llm_reduce(accumulator:List[Optional[str]], model:str, llm:OpenAI, query:str) -> Optional[str]:
    logger.info('reduce phase')
    accumulator = [ res for res in accumulator if res is not None ]

    if len(accumulator) == 0:
        return None 
    
    context = []
    for index, segment in enumerate(accumulator):
        context.append(
            f"""segment number {index}:
                {segment}
            """
        )
    
    context = "\n###\n".join(context)
    out_reduce:ChatCompletion = llm.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": reduce_system_prompt
            }, 
            {
                "role": "user",                 
                "content": f"""Query: {query}
                segments : 
                {context}
                Combine these segments into a unified response that addresses the query. Maintain all relevant information while eliminating redundancies.
                """
            }
        ],
        max_tokens=1024
    )
    stringyfied_data = out_reduce.choices[0].message.content
    return stringyfied_data


def llm_map_reduce(query:str, model:str, llm:OpenAI, context_size:int, pages:List[str]) -> Optional[str]:
    if len(pages) == 0:
        return None  
    
    if len(pages) < context_size:
        page = "\n".join(pages)
        return llm_map(
            page=page, model=model, llm=llm, query=query
        )
    
    nb_pages = len(pages)
    batch_size = int(nb_pages / context_size)
    paritions = []
    for counter in range(0, nb_pages, batch_size):
        paritions.append(
            pages[counter:counter+batch_size]
        )

    accumulator:List[Optional[str]] = []
    with ThreadPoolExecutor(max_workers=len(paritions)) as executor:
        fn = partial(llm_map_reduce, query, model, llm, context_size)
        features = executor.map(fn, paritions)
        for map_response in features:
            accumulator.append(map_response)

    out_response = llm_reduce(
        query=query, model=model, llm=llm, accumulator=accumulator
    )
    return out_response

