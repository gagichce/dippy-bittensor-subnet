import hashlib
import os
import functools
import subprocess
import shutil
import multiprocessing
from typing import Any, Tuple
import bittensor as bt
import requests
from transformers import AutoModelForCausalLM, AutoConfig

from model.data import ModelId


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters."
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Hugging Face repo id must be in the format <org or user name>/<repo_name>."
        )

    return parts[0], parts[1]


def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        try:
            result = func()
            queue.put(result)
        except (Exception, BaseException) as e:
            # Catch exceptions here to add them to the queue.
            queue.put(e)

    # Use "fork" (the default on all POSIX except macOS), because pickling doesn't seem
    # to work on "spawn".
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


def regenerate_hash(namespace, name, chat_template, competition_id):
    s = " ".join([namespace, name, chat_template, competition_id])
    hash_output = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return int(hash_output[:16], 16)  # Returns a 64-bit integer from the first 16 hexadecimal characters


def save_model(model, tokenizer, path: str, model_name: str):
    """Saves a model and tokenizer to a path.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        path (str): The path to save the model and tokenizer to.
    """
    # make sure the path exists
    os.makedirs(path, exist_ok=True)

    model_path = os.path.join(path, model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
<<<<<<< Updated upstream
    bt.logging.success(f"Model and tokenizer saved to {path}")
=======
    bt.logging.success(f"Model and tokenizer saved to {path}")

def load_model_no_download(repo_namespace: str, repo_name: str):
    """
    Validate the model by loading it, without downloading it from the Hugging Face Hub
    """
    try:
        config = AutoConfig.from_pretrained('/'.join([repo_namespace, repo_name]), revision='main')
    except Exception as e:
        return None, str(e)
    
    if config is not None:
        print("Model configuration retrieved from Hub")
        try:
            # Check if GPU is available and see if model fits in GPU
            print('loading model in RAM to check if it fits in GPU')
            model = AutoModelForCausalLM.from_config(
                config=config,
            )
            print("Model loaded successfully")
            return model, None
        except Exception as e:
            return None, str(e)
    else:
        return None, "Could not retrieve model configuration from Hub"
    

def parse_size(line):
    """
    Parse the size string with unit and convert it to bytes.
    
    Args:
    - size_with_unit (str): The size string with unit (e.g., '125 MB')
    
    Returns:
    - int: The size in bytes
    """
    try:
        # get number enclosed in brackets
        size, unit = line[line.find("(")+1:line.rfind(")")].strip().split(' ')
        size = float(size.replace(',', ''))  # Remove commas for thousands
        unit = unit.lower()
        if unit == 'kb':
            return int(size * 1024)
        elif unit == 'mb':
            return int(size * 1024 * 1024)
        elif unit == 'gb':
            return int(size * 1024 * 1024 * 1024)
        else:
            raise ValueError(f"Unknown unit: {unit}")
    except ValueError as e:
        print(f"Error parsing size string '{size}{unit}': {e}")
        return 0


def check_model_repo_size(hash: int, repo_namespace: str, repo_name: str) -> int:
    """
    Check the size of a model hosted on Hugging Face using Git LFS without checking out the files,
    and clean up the cloned repository afterwards, even if an error occurs.
    
    Args:
    - hash (int): The hash of the model
    - repo_namespace (str): The namespace of the model repository
    - repo_name (str): The name of the model repository
    
    Returns:
    - int: The total size of the model files in bytes
    """
    repo_dir = f"data/{str(hash)}/models--{repo_namespace}--{repo_name}"
    original_dir = os.getcwd()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.run(["git", "clone", "--no-checkout", f"https://huggingface.co/{repo_namespace}/{repo_name}", repo_dir], check=True, timeout=10)
            os.chdir(repo_dir)
            lfs_files_output = subprocess.check_output(["git", "lfs", "ls-files", "-s"], text=True, timeout=10)
            total_size = sum(parse_size(line) for line in lfs_files_output.strip().split('\n') if line)
            return total_size
        except subprocess.TimeoutExpired as e:
            print(f"Operation timed out on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("Max retries exceeded.")
                return None
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("Max retries exceeded.")
                return None
        finally:
            os.chdir(original_dir)
            shutil.rmtree(os.path.join(original_dir, repo_dir), ignore_errors=True)


def get_model_size(repo_namespace: str, repo_name: str):
    safetensor_index = f"https://huggingface.co/{repo_namespace}/{repo_name}/resolve/main/model.safetensors.index.json"
    response = requests.get(safetensor_index)
    if response.status_code != 200:
        print(f"Error getting safetensors index: {response.text}")
        return None
    
    response_json = response.json()
    if 'metadata' not in response_json:
        print("Error: metadata not found in safetensors index")
        return None
    
    if 'total_size' not in response_json['metadata']:
        print("Error: total_size not found in safetensors index metadata")
        return None
    
    total_size = response_json['metadata']['total_size']
    
    return total_size
>>>>>>> Stashed changes
