"""
API封装实战 - Week 4 部署优化

本模块涵盖大模型API封装的完整实现，包括：
1. FastAPI基础配置与路由设计
2. RESTful接口实现
3. 异步请求处理与批处理优化
4. 请求验证与错误处理
5. API文档自动生成
6. 中间件与限流策略
7. 多模型管理与负载均衡

Author: learn_llm
"""

import torch
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Tuple, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
import time
import uuid
import os
from pathlib import Path
from collections import defaultdict
from enum import Enum
import json
import hashlib
import redis
from prometheus_client import Counter, Histogram, start_http_server
from starlette.responses import Response
import io
from PIL import Image
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelInfo:
    model_name: str
    model_path: str
    status: ModelStatus = ModelStatus.LOADING
    device: str = "cuda"
    max_batch_size: int = 8
    current_requests: int = 0
    loaded_at: Optional[datetime] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    do_sample: Optional[bool] = True

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError('messages cannot be empty')
        return v


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class TokenizeRequest(BaseModel):
    model: str
    text: str
    return_tensors: Optional[bool] = False


class TokenizeResponse(BaseModel):
    id: str
    object: str = "tokenize"
    model: str
    tokens: List[int]
    token_strings: List[str]


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "list"
    model: str
    data: List[Dict[str, Any]]


class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]]
    batch_size: Optional[int] = Field(default=4, ge=1, le=32)


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    status: str
    results: List[Dict[str, Any]]
    processing_time: float


class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.model_cache: Dict[str, Any] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.load_balancer_strategy: str = "round_robin"
        self.current_index: int = 0

    def init_redis(self, host: str = "localhost", port: int = 6379):
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            logger.info(f"Redis connected to {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    def register_model(self, model_name: str, model_path: str, device: str = "cuda", max_batch_size: int = 8):
        model_info = ModelInfo(
            model_name=model_name,
            model_path=model_path,
            device=device,
            max_batch_size=max_batch_size
        )
        self.models[model_name] = model_info
        logger.info(f"Model registered: {model_name}")

    def load_model(self, model_name: str) -> bool:
        if model_name not in self.models:
            return False

        model_info = self.models[model_name]
        try:
            if model_info.device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if model_name in self.model_cache:
                return True

            logger.info(f"Loading model {model_name} to {device}")
            model_info.status = ModelStatus.LOADING

            if "llama" in model_name.lower() or "qwen" in model_name.lower():
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_info.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_info.model_path)
                self.model_cache[f"{model_name}_model"] = model
                self.model_cache[f"{model_name}_tokenizer"] = tokenizer
            else:
                from transformers import AutoModel, AutoTokenizer
                model = AutoModel.from_pretrained(model_info.model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_info.model_path)
                model = model.to(device)
                self.model_cache[f"{model_name}_model"] = model
                self.model_cache[f"{model_name}_tokenizer"] = tokenizer

            model_info.status = ModelStatus.READY
            model_info.loaded_at = datetime.now()
            logger.info(f"Model {model_name} loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            model_info.status = ModelStatus.ERROR
            return False

    def get_model(self, model_name: str) -> Optional[Any]:
        if model_name not in self.models:
            return None
        return self.model_cache.get(f"{model_name}_model")

    def get_tokenizer(self, model_name: str) -> Optional[Any]:
        if model_name not in self.models:
            return None
        return self.model_cache.get(f"{model_name}_tokenizer")

    def select_model_for_inference(self, model_names: List[str]) -> str:
        available_models = [
            name for name in model_names
            if name in self.models
            and self.models[name].status == ModelStatus.READY
            and self.models[name].current_requests < self.models[name].max_batch_size
        ]

        if not available_models:
            if model_names and model_names[0] in self.models:
                return model_names[0]
            return None

        if self.load_balancer_strategy == "round_robin":
            selected = available_models[self.current_index % len(available_models)]
            self.current_index += 1
            return selected

        elif self.load_balancer_strategy == "least_requests":
            return min(available_models, key=lambda x: self.models[x].current_requests)

        return available_models[0]

    def check_rate_limit(self, api_key: str, max_requests: int = 100, window_seconds: int = 60) -> Tuple[bool, int]:
        if not self.redis_client:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=window_seconds)
            self.request_counts[api_key] = [
                t for t in self.request_counts[api_key]
                if t > window_start
            ]
            remaining = max_requests - len(self.request_counts[api_key])
            if len(self.request_counts[api_key]) >= max_requests:
                return False, 0
            self.request_counts[api_key].append(current_time)
            return True, remaining

        try:
            key = f"rate_limit:{api_key}"
            current = self.redis_client.incr(key)
            if current == 1:
                self.redis_client.expire(key, window_seconds)
            remaining = max(max_requests - current, 0)
            if current > max_requests:
                return False, 0
            return True, remaining
        except Exception:
            return True, 100


model_manager = ModelManager()


class APISecurity:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("API_SECRET_KEY", "default-secret-key")
        self.valid_api_keys: Dict[str, Dict] = {}

    def validate_api_key(self, api_key: str) -> bool:
        if not api_key:
            return False
        return api_key in self.valid_api_keys or self._generate_api_key_hash(api_key) in self.valid_api_keys

    def _generate_api_key_hash(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        api_key = f"sk-{uuid.uuid4().hex}"
        key_hash = self._generate_api_key_hash(api_key)
        self.valid_api_keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions or ["inference"],
            "created_at": datetime.now().isoformat()
        }
        return api_key

    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> Dict:
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing authorization")
        api_key = credentials.credentials
        key_hash = self._generate_api_key_hash(api_key)
        if key_hash not in self.valid_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return self.valid_api_keys[key_hash]


security = APISecurity()
auth_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)
) -> Dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    return security.verify_token(credentials)


class InferenceEngine:
    @staticmethod
    async def chat_completion(
        model_name: str,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        do_sample: bool = True
    ) -> Union[Dict, AsyncGenerator[str, None]]:
        model = model_manager.get_model(model_name)
        tokenizer = model_manager.get_tokenizer(model_name)

        if not model or not tokenizer:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model_info = model_manager.models[model_name]
        model_info.current_requests += 1

        try:
            if stream:
                return InferenceEngine._stream_chat_completion(
                    model, tokenizer, messages, max_tokens, temperature, top_p, do_sample
                )
            else:
                return await InferenceEngine._sync_chat_completion(
                    model, tokenizer, messages, max_tokens, temperature, top_p, do_sample
                )
        finally:
            model_info.current_requests -= 1

    @staticmethod
    async def _sync_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> Dict:
        start_time = time.time()

        conversation = ""
        for msg in messages:
            conversation += f"{msg.role}: {msg.content}\n"

        inputs = tokenizer(conversation, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text.replace(conversation, "").strip()

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(datetime.now().timestamp())

        input_tokens = len(inputs["input_ids"][0])
        output_tokens = len(outputs[0]) - input_tokens

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_manager.models.get(type(model).__name__, type(model).__name__),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }

    @staticmethod
    async def _stream_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> AsyncGenerator[str, None]:
        conversation = ""
        for msg in messages:
            conversation += f"{msg.role}: {msg.content}\n"

        inputs = tokenizer(conversation, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(datetime.now().timestamp())

        generated_text = ""
        with torch.no_grad():
            for i, token_id in enumerate(model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                return_dict_in_generate=True,
                output_scores=True
            )):
                if i >= max_tokens:
                    break

                new_token = tokenizer.decode(token_id[0], skip_special_tokens=True)
                generated_text += new_token

                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": type(model).__name__,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_token},
                        "finish_reason": None if i < max_tokens - 1 else "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    @staticmethod
    def get_embeddings(
        model_name: str,
        input_text: Union[str, List[str]],
        encoding_format: str = "float",
        dimensions: Optional[int] = None
    ) -> Dict:
        model = model_manager.get_model(model_name)
        tokenizer = model_manager.get_tokenizer(model_name)

        if not model or not tokenizer:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        if isinstance(input_text, str):
            input_text = [input_text]

        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        if encoding_format == "base64":
            embeddings = [base64.b64encode(e.tobytes()).decode() for e in embeddings]

        if dimensions and dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :dimensions]

        response_id = f"embd-{uuid.uuid4().hex}"

        return {
            "id": response_id,
            "object": "list",
            "model": model_name,
            "data": [{
                "object": "embedding",
                "index": i,
                "embedding": emb.tolist() if encoding_format == "float" else emb
            } for i, emb in enumerate(embeddings)]
        }


class BatchProcessor:
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False

    async def add_request(self, request: Dict) -> str:
        request_id = f"batch-{uuid.uuid4().hex}"
        await self.batch_queue.put((request_id, request))
        return request_id

    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        if not requests:
            return []

        results = []
        for i in range(0, len(requests), self.max_batch_size):
            batch = requests[i:i + self.max_batch_size]
            batch_results = await self._process_single_batch(batch)
            results.extend(batch_results)
        return results

    async def _process_single_batch(self, batch: List[Dict]) -> List[Dict]:
        await asyncio.sleep(0.01)
        return [{"id": req.get("id", f"res-{i}"), "status": "completed"} for i, req in enumerate(batch)]


batch_processor = BatchProcessor(max_batch_size=16)


app = FastAPI(
    title="LLM Serving API",
    description="High-performance LLM inference service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"Request {request_id}: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Request {request_id}: completed in {process_time:.4f}s")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    return response


@app.get("/v1/models")
async def list_models() -> Dict:
    return {
        "data": [
            {
                "id": name,
                "object": "model",
                "created": info.loaded_at.timestamp() if info.loaded_at else 0,
                "owned_by": "learn_llm"
            }
            for name, info in model_manager.models.items()
        ],
        "object": "list"
    }


@app.get("/v1/models/{model_name}")
async def get_model(model_name: str) -> Dict:
    if model_name not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    info = model_manager.models[model_name]
    return {
        "id": model_name,
        "object": "model",
        "created": info.loaded_at.timestamp() if info.loaded_at else 0,
        "owned_by": "learn_llm",
        "permission": [],
        "root": info.model_path,
        "parent": None
    }


@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    credentials: Dict = Depends(verify_api_key)
) -> Union[ChatCompletionResponse, StreamingResponse]:
    allowed, remaining = model_manager.check_rate_limit(credentials.get("user_id", "default"))

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Remaining": "0"}
        )

    if request.model not in model_manager.models:
        selected_model = model_manager.select_model_for_inference([request.model])
        if not selected_model:
            raise HTTPException(status_code=404, detail="Model not available")
        request.model = selected_model

    if not model_manager.models[request.model].loaded_at:
        loaded = model_manager.load_model(request.model)
        if not loaded:
            raise HTTPException(status_code=500, detail="Failed to load model")

    if request.stream:
        return StreamingResponse(
            InferenceEngine.chat_completion(
                model_name=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                do_sample=request.do_sample
            ),
            media_type="text/event-stream"
        )

    result = await InferenceEngine.chat_completion(
        model_name=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
        do_sample=request.do_sample
    )

    return ChatCompletionResponse(**result)


@app.post("/v1/embeddings")
async def create_embedding(
    request: EmbeddingRequest,
    credentials: Dict = Depends(verify_api_key)
) -> EmbeddingResponse:
    allowed, _ = model_manager.check_rate_limit(credentials.get("user_id", "default"))
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    result = InferenceEngine.get_embeddings(
        model_name=request.model,
        input_text=request.input,
        encoding_format=request.encoding_format,
        dimensions=request.dimensions
    )

    return EmbeddingResponse(**result)


@app.post("/v1/tokenize")
async def tokenize(
    request: TokenizeRequest,
    credentials: Dict = Depends(verify_api_key)
) -> TokenizeResponse:
    tokenizer = model_manager.get_tokenizer(request.model)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    tokens = tokenizer.encode(request.text)
    token_strings = tokenizer.convert_ids_to_tokens(tokens)

    return TokenizeResponse(
        id=f"tokn-{uuid.uuid4().hex}",
        object="tokenize",
        model=request.model,
        tokens=tokens,
        token_strings=token_strings
    )


@app.post("/v1/batch")
async def create_batch(
    batch_request: BatchRequest,
    credentials: Dict = Depends(verify_api_key)
) -> BatchResponse:
    start_time = time.time()

    results = await batch_processor.process_batch(batch_request.requests)

    processing_time = time.time() - start_time

    return BatchResponse(
        id=f"batch-{uuid.uuid4().hex}",
        object="batch",
        status="completed",
        results=results,
        processing_time=processing_time
    )


@app.get("/v1/batch/{batch_id}")
async def get_batch_status(batch_id: str) -> Dict:
    return {
        "id": batch_id,
        "status": "completed",
        "created_at": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check() -> Dict:
    cuda_available = torch.cuda.is_available()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda_available else 0

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu": {
            "available": cuda_available,
            "memory_gb": round(gpu_memory, 2)
        },
        "models_loaded": len(model_manager.model_cache) // 2,
        "models_registered": len(model_manager.models)
    }


@app.get("/metrics")
async def metrics() -> Dict:
    return {
        "total_requests": sum(
            len(times) for times in model_manager.request_counts.values()
        ),
        "models": {
            name: {
                "status": info.status.value,
                "requests": info.current_requests
            }
            for name, info in model_manager.models.items()
        }
    }


class APIDemo:
    @staticmethod
    def demo_api_endpoints():
        print("\n" + "=" * 60)
        print("API接口设计")
        print("=" * 60)

        print("\nRESTful API端点:")
        endpoints = [
            ("GET", "/v1/models", "列出所有可用模型"),
            ("GET", "/v1/models/{model_name}", "获取模型详情"),
            ("POST", "/v1/chat/completions", "对话补全"),
            ("POST", "/v1/embeddings", "生成向量嵌入"),
            ("POST", "/v1/tokenize", "分词编码"),
            ("POST", "/v1/batch", "批量请求处理"),
            ("GET", "/health", "健康检查"),
            ("GET", "/metrics", "性能指标")
        ]

        for method, path, description in endpoints:
            print(f"  {method:6} {path:35} {description}")

    @staticmethod
    def demo_request_format():
        print("\n" + "=" * 60)
        print("请求格式示例")
        print("=" * 60)

        print("\n对话补全请求:")
        print(json.dumps({
            "model": "qwen-7b-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }, indent=2))

        print("\n嵌入请求:")
        print(json.dumps({
            "model": "text-embedding-ada-002",
            "input": ["Hello world", "Machine learning"],
            "encoding_format": "float",
            "dimensions": 512
        }, indent=2))

    @staticmethod
    def demo_best_practices():
        print("\n" + "=" * 60)
        print("API设计最佳实践")
        print("=" * 60)

        practices = [
            "使用RESTful设计风格，语义清晰",
            "实现异步处理，支持高并发",
            "添加请求验证和错误处理",
            "使用流式响应优化用户体验",
            "实现限流和负载均衡",
            "添加完整的API文档",
            "使用认证保护API安全",
            "监控和指标收集"
        ]

        for i, practice in enumerate(practices, 1):
            print(f"  {i}. {practice}")


def demo_api_wrapper():
    print("=" * 60)
    print("API封装实战演示")
    print("=" * 60)

    APIDemo.demo_api_endpoints()
    APIDemo.demo_request_format()
    APIDemo.demo_best_practices()

    print("\n" + "=" * 60)
    print("API封装实战演示完成!")
    print("=" * 60)


def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    model_manager.register_model("qwen-7b-chat", "/path/to/qwen-7b-chat")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    demo_api_wrapper()
