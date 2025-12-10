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
        student_has_bos: bool = False,
        student_has_eos: bool = False,
        is_mlm: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_logprobs = num_logprobs
        self.teacher_temperature = teacher_temperature
        self.student_has_bos = student_has_bos
        self.student_has_eos = student_has_eos
        self.is_mlm = is_mlm

        self.loss = ChunkedTopKKDLoss(
            num_output_chunks=num_output_chunks, 
            kd_temperature=kd_temperature,
        )
        self.vllm_instance = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def get_teacher_forward(self, prompts: list, label_tokens: list):
        completion = self.vllm_instance.completions.create(
            model="/Users/nboizard/Downloads/bigemma3",
            prompt=self.tokenizer.batch_decode(prompts, skip_special_tokens=False),
            max_tokens=1,
            temperature=self.teacher_temperature,
            logprobs=self.num_logprobs,
            echo=True,
        )
        return self.format_vllm_logprobs(completion, label_tokens, renormalize=True)
    
    def format_vllm_logprobs(self, vllm_completion, label_tokens=None, renormalize=True):
        num_entries = 0
        for choice in vllm_completion.choices:
            valid_in_choice = sum(1 for lp in choice.prompt_logprobs if lp is not None)
            num_entries += int(self.student_has_bos) + self.is_mlm + valid_in_choice + int(self.student_has_eos)

        token_ids = torch.zeros((num_entries, self.num_logprobs), dtype=torch.long)
        token_logprobs = torch.zeros((num_entries, self.num_logprobs), dtype=torch.float32)
        masks = torch.ones((num_entries, self.num_logprobs), dtype=torch.bool)

        idx = 0
        label_idx = 0
        label_limit = len(label_tokens) if label_tokens is not None else 0

        for choice in vllm_completion.choices:
            if self.student_has_bos:
                masks[idx, :] = False
                idx += 1
            if self.is_mlm:
                masks[idx, :] = False
                idx += 1

            for logprobs in choice.prompt_logprobs:
                if logprobs is None:
                    continue
                
                items = list(logprobs.items())[:self.num_logprobs]
                current_k = len(items)

                if current_k > 0:
                    c_ids, c_vals = zip(*items)

                    token_ids[idx, :current_k] = torch.tensor([int(k) for k in c_ids], dtype=torch.long)
                    token_logprobs[idx, :current_k] = torch.tensor([v['logprob'] for v in c_vals], dtype=torch.float32)

                if label_idx < label_limit:
                    if label_tokens[label_idx] == -100:
                        masks[idx, :] = False
                    label_idx += 1
                
                idx += 1

            if self.student_has_eos:
                masks[idx, :] = False
                idx += 1

        if renormalize:
            row_log_z = torch.logsumexp(token_logprobs, dim=-1, keepdim=True)
            token_logprobs -= row_log_z
            token_logprobs[~masks] = 0.0

        return token_ids, token_logprobs, masks

    @staticmethod
    def _is_teacher_use_bos_eos(tokenizer):
        """Check if the tokenizer uses BOS and EOS tokens.
        
        returns: (has_bos: bool, has_eos: bool)
        """
        tokens = tokenizer.encode("")
        
        has_bos = tokenizer.bos_token_id in tokens
        has_eos = tokenizer.eos_token_id in tokens
        return has_bos, has_eos

    # def format_vllm_logprobs(self, vllm_completion, label_tokens=None, renormalize=True):
    #     """
    #     Format VLLM log probabilities into structured tensors.

    #     Args:
    #         vllm_completion: VLLM completion object containing choices with prompt_logprobs
    #         num_logprobs (int): Expected number of log probabilities per token
    #         renormalize (bool): Whether to renormalize logprobs to ensure they sum to 1

    #     Returns:
    #         tuple: (token_ids, token_logprobs, masks) - three tensors of shape [N, num_logprobs]
    #             where N is the total number of valid tokens across all choices
    #     """
    #     num_valid_tokens = sum(
    #         1 for choice in vllm_completion.choices
    #         for logprobs in choice.prompt_logprobs
    #         if logprobs is not None
    #     )
        
    #     token_ids = torch.empty((num_valid_tokens, self.num_logprobs), dtype=torch.long)
    #     token_logprobs = torch.empty((num_valid_tokens, self.num_logprobs), dtype=torch.float32)
        
    #     idx = 0
    #     for choice in vllm_completion.choices:
    #         for logprobs in choice.prompt_logprobs:
    #             if logprobs is None:
    #                 continue
                
    #             items = list(logprobs.items())[:self.num_logprobs]
                
    #             token_ids[idx, :len(items)] = torch.tensor(
    #                 [int(k) for k, _ in items], dtype=torch.long
    #             )
    #             token_logprobs[idx, :len(items)] = torch.tensor(
    #                 [v['logprob'] for _, v in items], dtype=torch.float32
    #             )
                
    #             if renormalize:
    #                 logZ = torch.logsumexp(token_logprobs[idx, :len(items)], dim=-1, keepdim=True)
    #                 token_logprobs[idx, :len(items)] -= logZ
                
    #             idx += 1
        
    #     masks = torch.ones((num_valid_tokens, self.num_logprobs), dtype=torch.bool)
        
    #     return token_ids, token_logprobs, masks