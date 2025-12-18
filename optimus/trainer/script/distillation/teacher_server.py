import argparse
import logging
import uvicorn
import torch
import time
import torch.nn.functional as F
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from fastapi.responses import JSONResponse

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

parser = argparse.ArgumentParser(description="Teacher Distillation Server (compatible with HuggingFace Accelerate)")
parser.add_argument("model", type=str, help="Path to teacher model")
parser.add_argument("--port", type=int, default=8000, help="Server port")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("distill-server")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"Loading model: {args.model} | Device: {self.device}")
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.device == "cuda" and FLASH_ATTN_AVAILABLE else "sdpa"
        )
        self.model.eval()
        self.model.config.use_cache = False

        if self.device == "cuda":
            logger.info("Compiling model forward pass (reduce-overhead)...")
            self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

        logger.info("Warming up...")
        try:
            dummy = torch.zeros((2, 16), dtype=torch.long, device=self.model.device)
            with torch.inference_mode():
                self.model(dummy, use_cache=False)
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")
            
        logger.info("Model Ready.")

engine = InferenceEngine()

class CompletionRequest(BaseModel):
    prompt: List[List[int]]
    logprobs: Optional[int] = 5

class CompletionResponse(BaseModel):
    token_ids_list: List[List[List[int]]]
    token_logprobs_list: List[List[List[float]]]

def blocking_inference(req: CompletionRequest):
    model = engine.model
    
    batch_size = len(req.prompt)
    lengths = [len(p) for p in req.prompt]
    max_len = max(lengths)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=model.device)
    
    for i, seq in enumerate(req.prompt):
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=model.device)
        input_ids[i, :lengths[i]] = seq_tensor

    start = time.time()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False
        )
    end = time.time()
    logger.info(f"Inference | Batch: {batch_size} | Max Len: {max_len} | Time: {end - start:.2f}s")

    all_logits = outputs.logits
    
    shift_logits = all_logits[:, :-1, :]  
    shift_labels = input_ids[:, 1:]       

    log_probs = F.log_softmax(shift_logits, dim=-1)

    chosen_token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    k = req.logprobs if req.logprobs else 1
    top_vals, top_inds = torch.topk(log_probs, k, dim=-1)

    label_in_topk_mask = (top_inds == shift_labels.unsqueeze(-1)).any(dim=-1)
    missing_indices_mask = ~label_in_topk_mask
    
    if missing_indices_mask.any():
        top_inds[missing_indices_mask, -1] = shift_labels[missing_indices_mask]
        top_vals[missing_indices_mask, -1] = chosen_token_logprobs[missing_indices_mask]

    top_inds_cpu = top_inds.tolist()
    top_vals_cpu = top_vals.tolist()
    
    final_ids_list = []
    final_vals_list = []

    for i, original_len in enumerate(lengths):
        valid_len = original_len - 1
        
        if valid_len > 0:
            final_ids_list.append(top_inds_cpu[i][:valid_len])
            final_vals_list.append(top_vals_cpu[i][:valid_len])
        else:
            final_ids_list.append([])
            final_vals_list.append([])

    return final_ids_list, final_vals_list

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/logprobs", response_model=CompletionResponse)
async def score_sequence(req: CompletionRequest):
    final_ids, final_vals = await run_in_threadpool(blocking_inference, req)
    
    return CompletionResponse(
        token_ids_list=final_ids,
        token_logprobs_list=final_vals
    )

@app.get("/health")
async def health_check():
    if engine.model is None:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready", "model": args.model}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="error",
    )