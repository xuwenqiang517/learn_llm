
from fastapi import FastAPI
from pydantic import BaseModel
try:
    from transformers import pipeline
    gen = pipeline('text-generation', model='distilgpt2')
except Exception:
    gen = None

app = FastAPI()

class Req(BaseModel):
    prompt: str

@app.post('/generate')
def generate(req: Req):
    if gen is None:
        return {'error': '模型未加载，检查依赖或网络'}
    out = gen(req.prompt, max_length=50, num_return_sequences=1)
    return {'text': out[0]['generated_text']}
