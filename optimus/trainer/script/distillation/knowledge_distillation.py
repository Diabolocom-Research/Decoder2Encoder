import torch
from openai import OpenAI

from optimus.trainer.script.distillation.kullback_leibler_divergence import ChunkedTopKKDLoss


class KnowledgeDistillation:
    """
    Knowledge Distillation using Top-K KL Divergence Loss.
    """

    def __init__(
        self,
        tokenizer,
        num_logprobs=512,
        num_output_chunks: int = 8,
        kd_temperature: int = 1,
        teacher_temperature: float = 0,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
    ):
        self.tokenizer = tokenizer
        self.num_logprobs = num_logprobs
        self.teacher_temperature = teacher_temperature

        self.loss = ChunkedTopKKDLoss(num_output_chunks=num_output_chunks, kd_temperature=kd_temperature)
        self.vllm_instance = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def get_teacher_forward(self, prompts: list, label_token_ids: list):
        completion = self.vllm_instance.completions.create(
            model="Qwen/Qwen3-0.6B-Base",
            prompt=self.tokenizer.batch_decode(prompts, skip_special_tokens=False),
            max_tokens=1,
            temperature=self.teacher_temperature,
            logprobs=self.num_logprobs,
            echo=True,
        )
        return self.format_vllm_logprobs(completion, renormalize=True)
    
    def format_vllm_logprobs(self, vllm_completion, renormalize=True):
        """
        Format VLLM log probabilities into structured tensors.

        Args:
            vllm_completion: VLLM completion object containing choices with prompt_logprobs
            num_logprobs (int): Expected number of log probabilities per token
            renormalize (bool): Whether to renormalize logprobs to ensure they sum to 1

        Returns:
            tuple: (token_ids, token_logprobs, masks) - three tensors of shape [N, num_logprobs]
                where N is the total number of valid tokens across all choices
        """
        num_valid_tokens = sum(
            1 for choice in vllm_completion.choices
            for logprobs in choice.prompt_logprobs
            if logprobs is not None
        )
        
        token_ids = torch.empty((num_valid_tokens, self.num_logprobs), dtype=torch.long)
        token_logprobs = torch.empty((num_valid_tokens, self.num_logprobs), dtype=torch.float32)
        
        idx = 0
        for choice in vllm_completion.choices:
            for logprobs in choice.prompt_logprobs:
                if logprobs is None:
                    continue
                
                items = list(logprobs.items())[:self.num_logprobs]
                
                token_ids[idx, :len(items)] = torch.tensor(
                    [int(k) for k, _ in items], dtype=torch.long
                )
                token_logprobs[idx, :len(items)] = torch.tensor(
                    [v['logprob'] for _, v in items], dtype=torch.float32
                )
                
                if renormalize:
                    logZ = torch.logsumexp(token_logprobs[idx, :len(items)], dim=-1, keepdim=True)
                    token_logprobs[idx, :len(items)] -= logZ
                
                idx += 1
        
        masks = torch.ones((num_valid_tokens, self.num_logprobs), dtype=torch.bool)
        return token_ids.tolist(), token_logprobs.tolist(), masks.tolist()