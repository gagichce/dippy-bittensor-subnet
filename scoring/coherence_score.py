import os
import random

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import ray
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import huggingface_hub
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from accelerate import PartialState
from typing import List

# prompt dataset: ~256 prompts

# Import necessary modules and functions from the main API file
from scoring.common import (
    full_path,
    EvaluateModelRequest,
    PROMPTS_1_FILENAME,
    MAX_SEQ_LEN_VIBE_SCORE,
    MAX_GENERATION_LENGTH,
    PROMPTS_2_FILENAME,
    chat_template_mappings,
    SAMPLE_SIZE_VIBE_SCORE,
    BATCH_SIZE_VIBE_SCORE,
    MAX_GENERATION_LEEWAY,
    PIPPA_FILENAME,
    COHERENCE_SAMPLE_SIZE,
    COHERENCE_MAX_TOKENS,
    COHERENCE_EVAL_MODEL,
    COHERENCE_NUM_EVALS,
    VLLM_GPU_MEMORY,
)
from scoring.dataset import PromptDataset, PippaDataset, CoherenceDataset

coherence_dataset = CoherenceDataset(
    dataset_id="AlekseyKorshuk/synthetic-friendly-characters",
    split_id="train",
    max_input_len=MAX_SEQ_LEN_VIBE_SCORE - MAX_GENERATION_LENGTH - 200,
)

# TODO: Replace with corcel
from openai import OpenAI

remote_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def coherence_evaluator(generated_text: dict[str, str]):
    evaluation_text = f'''
    You are a text coherence analyzer.
    Your task is to assess the coherence of the following text.
    Coherent text should have logical flow, clear connections between ideas, and maintain a consistent theme or purpose throughout.
    The text may cut off due to a hardcoded limit. If this is the case, judge the coherence of the text according to the output so far. 
    Respond only with:
    1 - if the text is coherent
    0 - if the text is not coherent

    Do not provide any explanation or additional output. Just respond with 1 or 0.

    The initial prompt is as follows. Do not grade this prompt, only use it as context when grading coherence:
    """
    {generated_text['input_text']}
    """
    The generated text is as follows. 
    
    """
    {generated_text['generated_text']}
    """
    Coherence assessment (1 or 0):
    '''

    print(evaluation_text)
    chat_completion = remote_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": evaluation_text,
            }
        ],
        model=COHERENCE_EVAL_MODEL,
    )
    score = int(chat_completion.choices[0].message.content)
    return score


def get_coherence_score(request: EvaluateModelRequest):
    try:
        input_tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}", revision=request.revision
        )
        # Set chat template params
        coherence_dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], input_tokenizer)

        # Unzip the sampled data
        chat_contexts, _ = zip(*coherence_dataset.sample_dataset(2048))

        model_name = f"{request.repo_namespace}/{request.repo_name}"
        print(f"calculate cohere score for {model_name} v1")
        cscore = calculate_coherence_score(
            model_name=model_name,
            revision=request.revision,
            chat_contexts=chat_contexts,
            tokenizer=input_tokenizer,
        )
        return {"coherence_score": cscore}
    except Exception as e:
        raise e

def calculate_coherence_score(model_name, revision, chat_contexts, tokenizer, verbose=False) -> int:

    # instantiate a vllm model as it is faster and more memory efficient for text generation
    model = LLM(
        model_name,
        revision=revision,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=VLLM_GPU_MEMORY,
        max_num_seqs=32,
        # max_num_seqs=10,
        max_model_len=MAX_SEQ_LEN_VIBE_SCORE,
        download_dir="/app/evalsets",
    )

    generated_samples = []

    # loop through the context in batches
    for i in range(0, len(chat_contexts), COHERENCE_SAMPLE_SIZE):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=COHERENCE_MAX_TOKENS,
        )
        prompts = chat_contexts[i : i + COHERENCE_SAMPLE_SIZE]
        outputs = model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        # Highly regarded technique
        for index, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            generated_sample = {
                "input_text": chat_contexts[i+index],
                "generated_text": generated_text,
            }
            generated_samples.append(generated_sample)

    coherence_score = 0

    # random.shuffle(generated_samples)
    coherence_mapping = {}
    bad_score = 0
    print(f"len(generated_samples)")
    print(len(generated_samples))
    for i in range(COHERENCE_NUM_EVALS):
        coherence_score = coherence_evaluator(generated_samples[i])
        coherence_mapping[i] = coherence_score
        if coherence_score < 1:
            bad_score += 1
            pass
    print("coherence mapping")
    print(coherence_mapping)
    fscore = bad_score / COHERENCE_NUM_EVALS
    print(f"coherence_full_score_fkk={fscore}")
    destroy_model_parallel()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    try:
        torch.distributed.destroy_process_group()
    except Exception as e:
        print("No process group to destroy")

    return coherence_score
