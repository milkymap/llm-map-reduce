import click 
from dotenv import load_dotenv

from src.log import logger 
from src.settings import Credentials

from typing import List, Tuple, Dict, Optional  
from openai import OpenAI

from PyPDF2 import PdfReader
from src.algorithms.strategies import llm_map_reduce

from dotenv import load_dotenv

@click.group(chain=False, invoke_without_command=True)
@click.pass_context
def handler(ctx:click.core.Context):
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj['settings'] = {
        'credentials': Credentials()
    } 

@handler.command()
@click.option('--path2file', '-p', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--model', '-m', type=click.Choice(choices=['gpt-4o-mini', 'gpt-4o']), default='gpt-4o-mini')
@click.option('--context_size', '-s', default=4, help='number of page for the map phase')
@click.option('--limit', '-l', default=32)
@click.pass_context
def map_reduce(ctx:click.core.Context, path2file:str, model:str, context_size:int, limit:int):
    credentials:Credentials = ctx.obj['settings']['credentials']
    llm = OpenAI(api_key=credentials.openai_api_key)

    pdf_reader = PdfReader(stream=path2file)
    pages:List[str] = [ page.extract_text() for page in pdf_reader.pages ]
    pages = [ page for page in pages if len(page) > 0]
    
    if len(pages) == 0:
        logger.warning('pdf file must contains at least one plain text page')
        exit(0)
    pages = pages[:limit]

    while True:
        try:
            query = input('query:')
            response:Optional[str] = llm_map_reduce(
                query=query,
                model=model,
                llm=llm,
                pages=pages,
                context_size=context_size
            )
            if response is None:
                logger.warning('none value was found during the map_reduce phase')
                continue
            print(response)
        except KeyboardInterrupt:
            break 
        except Exception as e:
            logger.error(e)
            break 


if __name__ == '__main__':
    load_dotenv()
    handler(obj={})