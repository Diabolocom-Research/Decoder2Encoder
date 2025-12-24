import time
import msgpack
from typing import List, Optional

import requests
import torch

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

        self.server_instance = LogprobsTeacherClient(
            base_url=self.train_config.kd_base_url,
            logger=logger,
        )
        self.loss = ChunkedTopKKDLoss(
            num_output_chunks=self.train_config.kd_num_output_chunks,
            kd_temperature=self.train_config.kd_temperature,
        )

        logger(f"Waiting for vLLM server at address: {self.train_config.kd_base_url}")
        self.server_instance.wait_for_ready(timeout=self.train_config.kd_server_timeout)
        logger("vLLM server reached.")
        logger(f"Knowledge Distillation initialized with Top-{self.train_config.kd_num_logprobs} KL Divergence Loss.")

    def get_teacher_forward(
        self, prompts: List[int], labels: Optional[torch.LongTensor], **kwargs
    ):
        if self.train_config.kd_teacher_skip_first_token:
            prompts = [p[1:] for p in prompts]

        completion = self.server_instance.get_logprobs(
            prompt=prompts, logprobs=self.train_config.kd_num_logprobs
        )

        return self.format_logprobs(
            completion,
            label_tokens=labels,
            slide_right_for_mlm=self.slide_right_for_mlm,
            skip_first_token=self.train_config.kd_teacher_skip_first_token,
            renormalize=True,
        )

    def format_logprobs(
        self,
        teacher_completion: dict[List[List[int]], List[List[float]]],
        label_tokens: Optional[List[int]] = None,
        slide_right_for_mlm=False,
        skip_first_token=False,
        renormalize=True,
    ):
        kd_num = self.train_config.kd_num_logprobs

        token_ids_flatten = []
        token_logprobs_flatten = []
        masks_flatten = []

        label_iter = iter(label_tokens) if label_tokens is not None else None

        for token_ids, token_logprobs in zip(
            teacher_completion["token_ids_list"],
            teacher_completion["token_logprobs_list"],
        ):
            if slide_right_for_mlm:
                token_ids_flatten, token_logprobs_flatten, masks_flatten = (
                    self._add_padding_token(
                        token_ids_flatten, token_logprobs_flatten, masks_flatten, kd_num
                    )
                )
            if skip_first_token:
                token_ids_flatten, token_logprobs_flatten, masks_flatten = (
                    self._add_padding_token(
                        token_ids_flatten, token_logprobs_flatten, masks_flatten, kd_num
                    )
                )

            token_ids_flatten.extend(token_ids)
            token_logprobs_flatten.extend(token_logprobs)

            if label_iter is None:
                masks_flatten.extend([[True] * kd_num] * len(token_ids))
                continue

            for _ in range(len(token_ids)):
                label = next(label_iter, None)
                is_valid = label != -100 if label is not None else True
                masks_flatten.append([is_valid] * kd_num)

        token_ids = torch.tensor(token_ids_flatten, dtype=torch.long)
        token_logprobs = torch.tensor(token_logprobs_flatten, dtype=torch.float32)
        masks = torch.tensor(masks_flatten, dtype=torch.bool)

        if renormalize:
            token_logprobs -= torch.logsumexp(token_logprobs, dim=-1, keepdim=True)

        return token_ids, token_logprobs, masks

    def _add_padding_token(
        self,
        token_ids_list: List,
        token_logprobs_list: List,
        masks_list: List,
        kd_num: int,
    ):
        token_ids_list.append([0] * kd_num)
        token_logprobs_list.append([-1e9] * kd_num)
        masks_list.append([False] * kd_num)
        return token_ids_list, token_logprobs_list, masks_list

class LogprobsTeacherClient:
    def __init__(self, base_url, logger):
        self.base_url = base_url.rstrip('/')
        self.logger = logger
        
        self.endpoint_url = self.base_url + "/logprobs"
        self.health_url = self.base_url + "/health"
        
        self.session = requests.Session()

    def wait_for_ready(self, timeout: int = 300, interval: int = 2):
        """
        Blocks until the server returns 200 OK on /health.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self.session.get(self.health_url, timeout=1)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            
            time.sleep(interval)
            
        raise TimeoutError(f"Server at {self.base_url} unreachable within {timeout} seconds.")

    def get_logprobs(self, prompt: List[List[int]], logprobs: Optional[int] = 5):
        """
        Direct simplified call matching your requested signature.
        """
        payload = msgpack.packb({"p": prompt, "k": logprobs})
        
        start = time.time()
        resp = self.session.post(
            self.endpoint_url, 
            data=payload,
            headers={"Content-Type": "application/msgpack"}
        )
        resp.raise_for_status()
        self.logger(f"Logprobs request took {time.time() - start:.2f} seconds.")
        
        return msgpack.unpackb(resp.content)