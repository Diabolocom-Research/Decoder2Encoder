import os
import sys
import time
import argparse
import logging
import threading
import itertools
import uuid
import asyncio
import concurrent.futures
from contextlib import asynccontextmanager

# Third-party imports
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import uvicorn
import msgpack
from fastapi import FastAPI, Request, Response
from transformers import AutoModelForCausalLM, logging as hf_logging

# --- 1. Global Performance Setup ---

# Enable uvloop for faster asyncio event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Global Torch Optimizations
torch.set_grad_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apply CUDA-specific optimizations only if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# --- 2. Configuration & Logging ---

def setup_logging():
    """Configures logging to be minimal in the main process."""
    logging.basicConfig(
        level=logging.WARNING, 
        format='%(asctime)s|%(message)s', 
        datefmt='%H:%M:%S'
    )
    # Silence external libraries
    hf_logging.set_verbosity_error()
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return logging.getLogger("server")

def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance LLM Logprob Server")
    parser.add_argument("model", type=str, help="Path or HuggingFace ID of the model")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--replicas", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Max dynamic batch size")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Thread pool size for serialization")
    return parser.parse_args()

# Initialize Config & Logger
ARGS = parse_args()
LOGGER = setup_logging()


# --- 3. GPU/MPS/CPU Worker Logic ---

def run_worker(replica_id, model_path, device_config, input_queue, output_queue, ready_counter):
    """
    Dedicated process for Inference.
    Handles memory allocation, batch/mask creation, inference, and shared memory return.
    """
    # Re-configure logging for this worker
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s|%(message)s', 
        datefmt='%H:%M:%S', 
        force=True
    )
    worker_logger = logging.getLogger(f"worker-{replica_id}")
    
    # Silence libs inside worker process
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    try:
        # A. Device Setup
        device_type = device_config['type']
        device_id = device_config['id']
        
        if device_type == "cuda":
            device = torch.device(f"cuda:{device_id}")
            # Dynamic memory map for CUDA splitting
            total_gpus = device_config['total_gpus']
            assigned_gpus = device_config['assigned_gpus']
            mem_map = {
                g: (int(torch.cuda.mem_get_info(g)[0] * 0.95) if g in assigned_gpus else 0) 
                for g in range(total_gpus)
            }
        elif device_type == "mps":
            device = torch.device("mps")
            mem_map = None # MPS manages unified memory differently
        else:
            device = torch.device("cpu")
            mem_map = None

        # B. Dtype Selection
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device_type == "cpu" and torch.cuda.is_bf16_supported(): # Check for AVX512_BF16 roughly
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # C. Model Loading
        # Note: device_map="auto" is primarily for CUDA. For MPS/CPU we trust the device object placement.
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "dtype": dtype,
            "trust_remote_code": True,
        }

        if device_type == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = mem_map
            load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # Fallback for MPS/CPU
            load_kwargs["device_map"] = None 
            
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        # Explicit move for non-CUDA if device_map didn't handle it
        if device_type != "cuda":
            model.to(device)

        model.eval()
        model.config.use_cache = False
        
        # Warmup
        model(torch.tensor([[0]*10], device=device))
        
        # --- HEALTH CHECK SIGNAL ---
        with ready_counter.get_lock():
            ready_counter.value += 1
        worker_logger.info(f"Worker {replica_id} READY.")

        # D. Inference Loop
        while True:
            item = input_queue.get()
            if item is None: 
                break
            
            req_id, flat_input_list, lengths_list, k = item
            start = time.time()

            # E. Optimized Batch Construction
            batch_size = len(lengths_list)
            max_len = max(lengths_list)
            
            # 1. Create Input IDs AND Attention Mask in system RAM (Fast CPU ops)
            input_ids_cpu = torch.zeros((batch_size, max_len), dtype=torch.long)
            attn_mask_cpu = torch.zeros((batch_size, max_len), dtype=torch.long)
            
            flat_tensor = torch.tensor(flat_input_list, dtype=torch.long)
            
            cursor = 0
            for i, l in enumerate(lengths_list):
                input_ids_cpu[i, :l] = flat_tensor[cursor:cursor+l]
                attn_mask_cpu[i, :l] = 1
                cursor += l
            
            # 2. Transfer (Non-blocking is hint, effective on CUDA)
            input_ids = input_ids_cpu.to(device, non_blocking=True)
            attn_mask = attn_mask_cpu.to(device, non_blocking=True)
            
            # F. Forward Pass
            out = model(
                input_ids, 
                attention_mask=attn_mask,
                use_cache=False, 
                output_attentions=False, 
                output_hidden_states=False
            )
            
            # Align logits (t) with labels (t+1)
            logits = out.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            
            # G. Top-K Selection & Label Injection
            top_v, top_i = torch.topk(log_probs, k, dim=-1)
            
            mask = (top_i == labels.unsqueeze(-1)).any(dim=-1)
            missing = ~mask
            if missing.any():
                chosen = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                top_i[missing, -1] = labels[missing]
                top_v[missing, -1] = chosen[missing]

            # H. Shared Memory Return
            cpu_i = top_i.detach().cpu()
            cpu_v = top_v.detach().cpu()
            
            cpu_i.share_memory_()
            cpu_v.share_memory_()
            
            output_queue.put((req_id, cpu_i, cpu_v, lengths_list, replica_id))

            # Metrics
            duration = time.time() - start
            worker_logger.info(f"Rep:{replica_id} | B:{batch_size} | {duration*1000:.1f}ms")
            
            # Cleanup
            del flat_input_list, lengths_list, flat_tensor, input_ids_cpu, attn_mask_cpu

    except Exception as e:
        worker_logger.error(f"Worker {replica_id} CRASHED: {e}")
        output_queue.put((None, None, None, None, -1))
        sys.exit(1)


# --- 4. Process Manager ---

class ProcessManager:
    """
    Orchestrates AsyncIO requests, Workers, and CPU serialization.
    """
    def __init__(self, config):
        self.config = config
        self.ctx = mp.get_context('spawn')
        
        # Queues
        self.worker_input_queues = []
        self.worker_output_queue = self.ctx.Queue()
        self.available_workers = asyncio.Queue()
        self.incoming_requests = asyncio.Queue()
        
        # State
        self.batch_map = {}
        self.loop = None
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.cpu_threads)
        
        # Health Check Shared Counter
        self.ready_counter = self.ctx.Value('i', 0)

    def start(self):
        # 1. Detect Environment
        if torch.cuda.is_available():
            device_type = "cuda"
            n_devices = torch.cuda.device_count()
            LOGGER.info(f"Starting in CUDA mode with {n_devices} GPUs.")
        elif torch.backends.mps.is_available():
            device_type = "mps"
            n_devices = 1
            LOGGER.info("Starting in MPS (Apple Silicon) mode.")
        else:
            device_type = "cpu"
            n_devices = 1
            LOGGER.info("Starting in CPU mode.")

        # 2. Spawn Workers
        per_rep = n_devices // self.config.replicas if device_type == "cuda" else 1
        
        for i in range(self.config.replicas):
            # Prepare Device Config
            dev_config = {'type': device_type}
            
            if device_type == "cuda":
                # Assign specific GPU slice to this worker
                gpus = list(range(i * per_rep, (i + 1) * per_rep))
                dev_config['id'] = gpus[0]
                dev_config['assigned_gpus'] = gpus
                dev_config['total_gpus'] = n_devices
            else:
                # MPS/CPU don't use IDs in the same way, usually just one shared device
                dev_config['id'] = 0 
                dev_config['assigned_gpus'] = []
                dev_config['total_gpus'] = 1

            q = self.ctx.Queue()
            self.worker_input_queues.append(q)
            
            p = self.ctx.Process(
                target=run_worker, 
                args=(i, self.config.model, dev_config, q, self.worker_output_queue, self.ready_counter), 
                daemon=True
            )
            p.start()
            self.available_workers.put_nowait(i)

        # 3. Start Loops
        self.loop = asyncio.get_running_loop()
        threading.Thread(target=self._result_listener, daemon=True).start()
        asyncio.create_task(self._scheduler())

    async def _scheduler(self):
        """Zero-Latency Greedy Scheduler."""
        while True:
            first_req = await self.incoming_requests.get()
            batch = [first_req]
            curr_count = len(first_req['p'])
            
            while curr_count < self.config.max_batch_size:
                try:
                    item = self.incoming_requests.get_nowait()
                    batch.append(item)
                    curr_count += len(item['p'])
                except asyncio.QueueEmpty:
                    break
            
            worker_id = await self.available_workers.get()
            self._dispatch_job(worker_id, batch)

    def _dispatch_job(self, worker_id, batch):
        batch_id = str(uuid.uuid4())
        max_k = max(x['k'] for x in batch)
        
        all_prompts = [p for x in batch for p in x['p']]
        flat_list = list(itertools.chain.from_iterable(all_prompts))
        lengths_list = [len(x) for x in all_prompts]

        meta = [{'fut': x['fut'], 'n': len(x['p']), 'k': x['k']} for x in batch]
        
        self.batch_map[batch_id] = (meta, max_k)
        self.worker_input_queues[worker_id].put((batch_id, flat_list, lengths_list, max_k))

    def _result_listener(self):
        """Thread listening to Shared Memory returns."""
        while True:
            bid, ids, vals, lens, wid = self.worker_output_queue.get()
            if ids is None: 
                self.loop.call_soon_threadsafe(self._kill_everything, wid)
            else:
                self.cpu_pool.submit(self._finalize_job, bid, ids, vals, lens, wid)

    def _kill_everything(self, wid):
        LOGGER.critical(f"FATAL: Worker {wid} died. Initiating immediate system shutdown.")
        os._exit(1)

    def _finalize_job(self, bid, ids, vals, lens, wid):
        if bid not in self.batch_map: return
        meta_list, max_k = self.batch_map.pop(bid)
        
        cursor = 0
        for m in meta_list:
            n_seqs, k_req, fut = m['n'], m['k'], m['fut']
            end = cursor + n_seqs
            
            user_lens = lens[cursor:end]
            chunk_ids = ids[cursor:end].tolist()
            chunk_vals = vals[cursor:end].tolist()
            
            res_ids, res_vals = [], []
            fast_path = (k_req == max_k)

            for i, length in enumerate(user_lens):
                valid_len = length - 1
                if valid_len > 0:
                    row_ids = chunk_ids[i][:valid_len]
                    row_vals = chunk_vals[i][:valid_len]
                    
                    if fast_path:
                        res_ids.append(row_ids)
                        res_vals.append(row_vals)
                    else:
                        res_ids.append([r[:k_req] for r in row_ids])
                        res_vals.append([r[:k_req] for r in row_vals])
                else:
                    res_ids.append([])
                    res_vals.append([])

            cursor += n_seqs
            
            packed_response = msgpack.packb({'token_ids_list': res_ids, 'token_logprobs_list': res_vals})
            self.loop.call_soon_threadsafe(fut.set_result, packed_response)

        self.loop.call_soon_threadsafe(self.available_workers.put_nowait, wid)


# --- 5. FastAPI Application ---

manager = ProcessManager(ARGS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    yield
    for q in manager.worker_input_queues:
        q.put(None)
    manager.cpu_pool.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    """Checks if all model replicas are loaded and ready."""
    if manager.ready_counter.value < ARGS.replicas:
        return Response(status_code=503, content="Initializing")
    return Response(status_code=200, content="Ready")

@app.post("/logprobs")
async def handle_logprobs(req: Request):
    body = await req.body()
    data = msgpack.unpackb(body)
    
    fut = asyncio.get_running_loop().create_future()
    manager.incoming_requests.put_nowait({
        'fut': fut, 
        'p': data['p'], 
        'k': data.get('k', 5)
    })
    
    return Response(content=await fut, media_type="application/msgpack")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="error")