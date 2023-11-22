from vllm.entrypoints.llm import LLM
from open_lm.model import Transformer
from transformers import PreTrainedModel
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.model import Transformer
import torch
from typing import Union, Tuple, Optional, List
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SamplerOutput
from vllm.model_executor.layers.sampler import Sampler, _apply_logits_processors, _get_output_tokens, _get_penalties, _apply_penalties, _get_temperatures, _get_top_p_top_k_min_p, _apply_top_p_top_k, _apply_min_p, _sample, _get_logprobs, _build_sampler_output
from vllm.model_executor.layers.sampler import _SAMPLING_EPS

class CustomSampler(Sampler):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)

    def forward(self, logits, input_metadata):
        """
        This is a copy-past of the forward method from vllm.model_executor.layers.sampler.Sampler
        with the unique difference that we pass in the logits instead of the hidden_state and the last layer weights.
        """

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, input_metadata)
        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties, repetition_penalties = (
            _get_penalties(input_metadata))
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        assert len(repetition_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, repetition_penalties)

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks, min_ps = _get_top_p_top_k_min_p(
            input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            logits = _apply_top_p_top_k(logits, top_ps, top_ks)

        do_min_p = any(mp > _SAMPLING_EPS for mp in min_ps)
        if do_min_p:
            logits = _apply_min_p(logits, min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, input_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, input_metadata, sample_results)
        return _build_sampler_output(sample_results, input_metadata,
                                     prompt_logprobs, sample_logprobs)


class OpenLMforCausalLM(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Transformer(config)
        self.lm_head = None
        # Initialize weights and apply final processing
        self.post_init()
        self.sampler = CustomSampler(config.vocab_size)

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        cache_dict = {"kv_caches": kv_caches, "input_metadata": input_metadata, "cache_events": cache_events}
        outputs = self.model(input_ids, positions, cache_dict)
        next_token = self.sampler(outputs.logits, outputs.hidden_state, input_metadata)
        return next_token

