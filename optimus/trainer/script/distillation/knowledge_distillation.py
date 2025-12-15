from typing import List, Optional
import time

import torch
from openai import OpenAI
from transformers import AutoTokenizer

from optimus.trainer.configuration.dataset import DatasetConfig
from optimus.trainer.configuration.train import TrainConfig
from optimus.trainer.script.distillation.kullback_leibler_divergence import (
    ChunkedTopKKDLoss,
)


class KnowledgeDistillation:
    """
    Knowledge Distillation using Top-K KL Divergence Loss.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        dataset_config: DatasetConfig,
        logger: callable,
    ):
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.slide_right_for_mlm = not self.train_config.mntp_objective

        self.vllm_instance = OpenAI(
            api_key=self.train_config.kd_api_key,
            base_url=self.train_config.kd_base_url,
        )
        self.loss = ChunkedTopKKDLoss(
            num_output_chunks=self.train_config.kd_num_output_chunks,
            kd_temperature=self.train_config.kd_temperature,
        )

        logger("Waiting for vLLM server.")
        self.is_server_available()
        logger("vLLM server reached.")

    async def get_teacher_forward(
        self, prompts: List[int], labels: Optional[torch.LongTensor], **kwargs
    ):
        if self.train_config.kd_teacher_skip_first_token:
            prompts = [p[1:] for p in prompts]

        completion = self.vllm_instance.completions.create(
            model=self.train_config.kd_teacher_name_or_path,
            prompt=prompts,
            max_tokens=1,
            temperature=self.train_config.kd_teacher_temperature,
            logprobs=self.train_config.kd_num_logprobs,
            echo=True,
        )
        return self.format_vllm_logprobs(
            completion,
            label_tokens=labels,
            slide_right_for_mlm=self.slide_right_for_mlm,
            skip_first_token=self.train_config.kd_teacher_skip_first_token,
            renormalize=True,
        )

    def format_vllm_logprobs(
        self,
        vllm_completion,
        label_tokens=None,
        slide_right_for_mlm=False,
        skip_first_token=False,
        renormalize=True,
    ):
        kd_num = self.train_config.kd_num_logprobs

        token_ids_list = []
        token_logprobs_list = []
        masks_list = []

        label_iter = iter(label_tokens) if label_tokens is not None else None

        for choice in vllm_completion.choices:
            if slide_right_for_mlm:
                token_ids_list, token_logprobs_list, masks_list = self._add_padding_token(token_ids_list, token_logprobs_list, masks_list, kd_num)
            if skip_first_token:
                token_ids_list, token_logprobs_list, masks_list = self._add_padding_token(token_ids_list, token_logprobs_list, masks_list, kd_num)

            for logprobs in choice.prompt_logprobs:
                if logprobs is None:
                    continue

                items = list(logprobs.items())[:kd_num]

                token_ids_list.append([int(k) for k, _ in items])
                token_logprobs_list.append([v["logprob"] for _, v in items])

                is_valid = True
                if label_iter:
                    try:
                        if next(label_iter) == -100:
                            is_valid = False
                    except StopIteration:
                        pass

                masks_list.append([is_valid] * kd_num)

        token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        token_logprobs = torch.tensor(token_logprobs_list, dtype=torch.float32)
        masks = torch.tensor(masks_list, dtype=torch.bool)

        if renormalize:
            token_logprobs -= torch.logsumexp(token_logprobs, dim=-1, keepdim=True)

        return token_ids, token_logprobs, masks

    def is_server_available(self, timeout=300):
        start = time.time()
        while True:
            try:
                self.vllm_instance.models.list()
                break
            except Exception:
                if time.time() - start > timeout:
                    exit()
                time.sleep(1)
                
    @staticmethod
    def _is_teacher_use_bos_eos(teacher_name_or_path):
        """Check if the tokenizer uses BOS and EOS tokens.

        returns: (has_bos: bool, has_eos: bool)
        """
        tokenizer = AutoTokenizer.from_pretrained(teacher_name_or_path)
        tokens = tokenizer.encode("")

        has_bos = tokenizer.bos_token_id in tokens
        has_eos = tokenizer.eos_token_id in tokens
        return has_bos, has_eos
    
    def _add_padding_token(self, token_ids_list, token_logprobs_list, masks_list, kd_num):
        token_ids_list.append([0] * kd_num)
        token_logprobs_list.append([0.0] * kd_num)
        masks_list.append([False] * kd_num)
        return token_ids_list, token_logprobs_list, masks_list