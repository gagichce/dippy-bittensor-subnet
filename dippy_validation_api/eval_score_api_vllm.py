import gc
import os
import time
from typing import Any
import tqdm
import shutil


from fastapi import FastAPI, HTTPException
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Import necessary modules and functions from the main API file
from dippy_validation_api.validation_api import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    PROB_TOP_K,
    SAMPLE_SIZE,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EvaluateModelRequest,
)

from dippy_validation_api.dataset import PippaDataset
from dippy_validation_api.validation_api import chat_template_mappings

app = FastAPI()

# create dir data if not exists
if not os.path.exists("data"):
    os.makedirs("data")
# download the file pippa_deduped.jsonl from huggingface
if not os.path.exists("data/pippa_deduped.jsonl"):
    huggingface_hub.hf_hub_download(repo_id="PygmalionAI/PIPPA", filename="pippa_deduped.jsonl", repo_type="dataset", local_dir = "data")

dataset = PippaDataset("data/pippa_deduped.jsonl", max_input_len=MAX_SEQ_LEN - MAX_GENERATION_LENGTH - 200)


def get_eval_score(
        model: LLM, 
        sampled_data: list[tuple], 
        tokenizer: AutoTokenizer, 
        request: EvaluateModelRequest,
        debug: bool = False
    ):
    """
    Evaluate the model on a dummy task
    """
    # maximum length this model can handle.
    max_len = MAX_SEQ_LEN
    
    # unzip the sampled data
    contexts, target_texts, _ = zip(*sampled_data)
    total_prob = 0
    count = 0

    # now we want to calculate the average probability of the target tokens that model assigns.
    batch_size = BATCH_SIZE
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(contexts), batch_size), desc="Evaluating batches"):
            # Tokenize the inputs and labels
            prompts = tokenizer(
                contexts[i:i+batch_size], 
                truncation=True,
                max_length=MAX_SEQ_LEN - MAX_GENERATION_LENGTH,
                add_special_tokens=False,
            )['input_ids']
            
            targets = tokenizer(
                target_texts[i:i+batch_size], 
                truncation=True,
                max_length=MAX_GENERATION_LENGTH,
                add_special_tokens=False,
            )['input_ids']

            full_context = []
            max_seq_len = 0
            for j in range(len(prompts['input_ids'])):
                if len(prompts['input_ids'][j]) + len(targets['input_ids'][j]) > max_len:
                    print("WARNING: Skipping sequence as it is too long")
                    continue
                full_context.append(prompts['input_ids'][j] + targets['input_ids'][j])   
                max_seq_len = max(max_seq_len, len(full_context[-1]))

            # Get model predictions (logits)
            try:
                if debug:
                    print("Getting model predictions for max sequence length: ", max_seq_len, " batch size: ", len(full_context))

                sampling_params = SamplingParams(
                    max_tokens=1,
                    prompt_logprobs=tokenizer.vocab_size,
                )
                model_output = model.generate(
                    prompt_token_ids=full_context, 
                    sampling_params=sampling_params,
                )
            except Exception as e:
                print("Error getting model predictions for max sequence length: ", max_seq_len, " batch size: ", len(full_context))
                raise ValueError("Error getting model predictions: " + str(e))

            batch_target_probs = []
            # Get the logits
            for j in range(len(model_output)):
                logits = model_output[j].prompt_logprobs
                target_probs = torch.tensor(len(targets[j]), tokenizer.vocab_size).cuda()
                for seq_index, probs in enumerate(logits[len(prompts[j]):]):
                    for token_index, token_obj in probs.items():
                        if token_obj.decoded_token == targets[j][seq_index]:
                            target_probs[seq_index][token_index] = token_obj.logprob
                
                if torch.isnan(target_probs).any():
                    raise ValueError("NaN values detected in the probabilities tensor")
                batch_target_probs.append(target_probs)
            

            if debug:
                # print the input tokens and top 10 predicted tokens
                print(f"Input: {tokenizer.decode(targets[0])}")
                for j in range(len(targets[0])):
                    actual_id = targets[0][j]
                    actual_token = tokenizer.decode([actual_id])
                    top_10_predicted_ids = torch.topk(batch_target_probs[0][j], 10).indices
                    top_10_predicted_tokens = [tokenizer.decode([id]) for id in top_10_predicted_ids]
                    print(f"Actual token: {actual_token}", f" -> top 10 pred tokens: {top_10_predicted_tokens}")

            
            # Convert list of variable length tensors to a padded tensor of probabilities and apply mask
            max_length = max(t.size(0) for t in batch_target_probs)
            padded_probs = [torch.nn.functional.pad(t, (0, 0, 0, max_length - t.size(0)), "constant", 0) for t in batch_target_probs]
            probabilities = torch.stack(padded_probs)
            target_mask = probabilities != 0
            probabilities[~target_mask] = 0

            # Get the top PROB_TOP_K indices and zero out all other probabilities
            top_prob_indices = torch.topk(probabilities, PROB_TOP_K, dim=-1).indices
            topk_mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, top_prob_indices, True)
            probabilities[~topk_mask] = 1e-9
            
            
            # Get the probabilities assigned by the model to the target tokens
            token_probabilities = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Mask out non target tokens
            token_probabilities = token_probabilities * target_mask

            # get the 1, 2, 3, 4 gram probabilities
            token_count = target_mask.sum().cpu().item()
            # 1-gram
            one_gram_probabilities = token_probabilities
            n_gram_prob = (one_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 2-gram
            two_gram_probabilities = one_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-1]
            n_gram_prob += (two_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 3-gram
            three_gram_probabilities = two_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-2]
            n_gram_prob += (three_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            # 4-gram
            four_gram_probabilities = three_gram_probabilities[:, 1:] * one_gram_probabilities[:, :-3]
            n_gram_prob += (four_gram_probabilities.sum().cpu().item() / token_count) * 0.25
            
            total_prob += n_gram_prob
            count += 1
            
            # delete the tensors to free up memory
            del model_output, batch_target_probs, padded_probs, probabilities, target_mask, token_probabilities
            del one_gram_probabilities, two_gram_probabilities, three_gram_probabilities, four_gram_probabilities
            del top_prob_indices, topk_mask

            gc.collect()
            torch.cuda.empty_cache()
    
    average_prob = total_prob / count
    print(f"Average probability of target tokens: {average_prob}")
    cleanup(model, True, request)

    return average_prob


def _prepare_dummy_input(model, device='cuda'):
    max_model_len = min(model.config.max_position_embeddings, MAX_SEQ_LEN)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, max_model_len), requires_grad=False, dtype=torch.int64, device=device)
    return input_ids

def warmup_model(model: LLM):
    """
    Warm up the model by running it on a dummy input
    """
    # run the max sequence length input through the model with batch size BATCH_SIZE
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            start_time = time()
            inputs = _prepare_dummy_input(model, device='cuda')
            start_time.record()
            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=0,
            )

            _ = model.generate(
                prompt_token_ids=inputs, 
                sampling_params=sampling_params,
            )
            end_time = time()

            latency = end_time - start_time # Measure latency in milliseconds
            latencies.append(latency)

        average_latency = sum(latencies) / len(latencies)
        print(f"Average model inference latency over 10 runs: {average_latency} ms")
        
    return average_latency

def cleanup(model, model_downloaded, request: EvaluateModelRequest):
    """
    Clean up the model data from memory and disk
    """
    # delete the model from memory
    with torch.no_grad():
        if model:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    total, used, _ = shutil.disk_usage("/")
    if used / total > 0.9:
        print("Warning: SSD is more than 90% full.") 
    if model_downloaded:
        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        hf_cache_info = huggingface_hub.scan_cache_dir()
        # delete from huggingface cache
        for repo_info in hf_cache_info.repos:
            revisions = repo_info.revisions
            if repo_info.repo_id == repo_id:
                for revision in revisions:
                    hf_cache_info.delete_revisions(revision.commit_hash)

@app.post("/eval_score")
def eval_score(request: EvaluateModelRequest):
    # get model size score
    try:
        print('Model weights downloaded successfully')
        temp_model = AutoModelForCausalLM.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
            revision=request.revision,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
        model_size = temp_model.get_memory_footprint()
        print('Model size: ', model_size, ' Bytes')
        print("Model number of parameters: ", model.num_parameters())
        # check if model size is within the limit. If not, return an error
        del temp_model
        gc.collect()
        if model_size > MAX_MODEL_SIZE:
            raise HTTPException(f"Model is too large when loaded in 4 bit quant: {model_size} Bytes. Should be less than {MAX_MODEL_SIZE} Bytes")
        
        model_size_score = 1 - (model_size / MAX_MODEL_SIZE)
        print('Model size score: ', model_size_score)
    except Exception as e:
        failure_reason = str(e)
        cleanup(None, model_downloaded, request)
        raise HTTPException("Error loading model: " + failure_reason)

    # Now download the weights
    print('Downloading model weights')
    model_downloaded = False
    failure_reason = ""
    # make dir data/hash if not exist
    if not os.path.exists(f"data/{str(request.hash)}"):
        os.makedirs(f"data/{str(request.hash)}")

    # get the tokenizers
    print('Downloading tokenizer')
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{request.repo_namespace}/{request.repo_name}",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # add a pad token if not present
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print('Tokenizer downloaded successfully')
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException("Error downloading tokenizer: " + failure_reason)

    try:
        model = LLM(
            model=f"{request.repo_namespace}/{request.repo_name}",
            revision=request.revision,
            max_context_len_to_capture=MAX_SEQ_LEN,
            seed=0,
            tensor_parallel_size=torch.cuda.device_count(),
            max_logprobs=tokenizer.vocab_size,
        )
    except Exception as e:
        raise HTTPException(f"Error loading model: {str(e)}")
        
    # warm up the model
    print('Warming up model')
    try:
        avg_latency = warmup_model(model)
        if not avg_latency: # either 0 or None
            raise HTTPException("Error warming up model")
            
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException("Error warming up model: " + failure_reason)

    # get latency score
    latency_score = 1 - (avg_latency / MAX_AVG_LATENCY)
    print('Latency score: ', latency_score)

    # set the chat template params
    dataset.set_chat_template_params(chat_template_mappings[request.chat_template_type], tokenizer)

    print('Sampling dataset')
    try:
        sampled_data = dataset.sample_dataset(SAMPLE_SIZE)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException("Error sampling dataset: " + failure_reason)
    
    # Part 2: Evaluate the model
    print('Evaluating model')
    try:
        eval_score = get_eval_score(
            model, 
            sampled_data, 
            tokenizer,
            request=request,
        )
        print('Model evaluation score: ', eval_score)
    except Exception as e:
        failure_reason = str(e)
        cleanup(model, model_downloaded, request)
        raise HTTPException("Error evaluating model: " + failure_reason)

    return {
        "eval_score": eval_score,
        "latency_score": latency_score,
        "model_size_score": model_size_score,
    }

@app.post("/shutdown")
def shutdown():
    print("Shutting down eval_score_api")
    os._exit(0)


if __name__ == "__main__":
    # The multiprocessing setup and uvicorn server run command will be similar to the main API file.
    import uvicorn
    # launch the api only if main process
    uvicorn.run(app, host="localhost", port=8001, timeout_keep_alive=960)
