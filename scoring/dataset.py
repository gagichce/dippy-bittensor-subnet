import json
import random
import jinja2

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import tiktoken


class PippaDataset(Dataset):
    def __init__(self, filename, max_input_len):
        self.filename = filename
        with open(filename, "r") as f:
            data = [json.loads(line) for line in f]

        self.dataset = self.process_data(data, max_input_len)

        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert pippa dataset to a format that can be used downstream.
        """
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # to get approx token count
        converted_dataset = []
        for data_point in data:
            # construct the system prompt using the bot_description and bot_greeting
            if not data_point["bot_definitions"]:
                data_point["bot_definitions"] = ""

            if not data_point["categories"]:
                data_point["categories"] = "None"

            system_prompt = f"""A chat between a user and a curious artificial intelligence that is an expert at roleplay. 
The AI is roleplaying as a character named {data_point['bot_name']}. 
The character's description: {data_point['bot_description']}. {data_point['bot_definitions']}.
The themes of the conversation are: {data_point['categories']}."""

            messages = [{"role": "system", "content": system_prompt}]

            messages.append(
                {
                    "role": "assistant",
                    "content": f"{data_point['bot_name']}: {data_point['bot_greeting']}",
                }
            )

            # get index of the last message from the chatbot
            last_message_index = 0
            input_len_so_far = len(encoding.encode(messages[0]["content"] + messages[1]["content"]))

            if input_len_so_far > max_input_len:
                # skip this data point
                continue

            for i, message in enumerate(data_point["conversation"]):
                input_len_so_far += len(encoding.encode(message["message"]))
                if input_len_so_far > max_input_len:
                    break

                if not message["is_human"]:
                    last_message_index = i

            last_user_message_index = 0
            for i, message in enumerate(data_point["conversation"][:last_message_index]):
                if message["is_human"]:
                    messages.append({"role": "user", "content": message["message"]})
                    last_user_message_index = i
                else:
                    messages.append({"role": "assistant", "content": f"{message['message']}"})

            character_response = data_point["conversation"][last_message_index]["message"]
            last_user_message = messages[last_user_message_index]["content"]

            converted_dataset.append(
                {
                    "messages": messages,
                    "last_user_message": last_user_message,  # get the last user message
                    "character_response": character_response,
                }
            )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._chat_template is None:
            raise ValueError("Chat template is not set. Please set the chat template before generating chat.")

        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set the tokenizer before generating chat.")

        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"

        return (
            chat_input,  # context
            f"{self.dataset[idx]['character_response']}{self._tokenizer.eos_token}",  # target text
            self.dataset[idx]["last_user_message"],  # last user message
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


class PromptDataset(Dataset):
    def __init__(self, filenames, max_input_len):
        all_data = []
        for filename in filenames:
            with open(filename, "r") as f:
                data = [json.loads(line) for line in f]
                all_data.append(data)

        self.dataset = self.process_data(all_data, max_input_len)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, data, max_input_len):
        """
        Convert opus dataset to a format that can be used downstream.
        """

        converted_dataset = []

        for data_point in data:
            for entry in data_point:
                # Always only 3 messages
                conversations = entry["conversations"]
                messages = [
                    {"role": "system", "content": conversations[0]["value"]},
                ]
                prompt_content = f'{conversations[1]["value"]} \n Please limit to (200-300) words.'
                messages.append(
                    {"role": "user", "content": prompt_content},
                )
                output = conversations[2]["value"]
                converted_dataset.append(
                    {
                        "messages": messages,
                        "output": output,
                    }
                )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._chat_template is None:
            raise ValueError("Chat template is not set. Please set the chat template before generating chat.")

        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set the tokenizer before generating chat.")
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"
        return (
            chat_input,  # context
            self.dataset[idx]["messages"],  # full message history
            self.dataset[idx]["output"],  # prompt output for comparison
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]


# import importlib
# import sys
# import os
#
# # Store the original sys.path
# original_sys_path = sys.path.copy()
#
# # Remove the directory containing your local 'datasets' folder from sys.path
# sys.path = [p for p in sys.path if p != 'datasets']
#
# # Import the 'datasets' package using importlib
# hf_datasets = importlib.import_module('datasets')
#
# # Restore the original sys.path
# sys.path = original_sys_path

from datasets import load_dataset
class CoherenceDataset(Dataset):
    def __init__(self, dataset_id: str, split_id: str, max_input_len):
        datass = load_dataset(dataset_id,split=split_id)

        self.dataset = self.process_data(datass, max_input_len)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, datass, max_input_len):
        """
        Convert dataset to a format that can be used downstream.
        """

        converted_dataset = []

        for entry in datass:
            system_prompt = f'''
            You are roleplaying a character with the name {entry['name']}.
            Your categories and topics of specialty are : {' '.join(entry['categories'])}.
            Your personality can be described with the following : {','.join(entry['personalities'])}.
            Your full character description:
            {entry['description']}
            '''
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            for message in entry['conversation']:
                role = "assistant" if message['role'] == 'character' else 'user'
                messages.append(
                {"role": role, "content": message['content']},
                )
            if messages[-1]["role"] == "assistant":
                del messages[-1]

            converted_dataset.append(
                {
                    "messages": messages,
                }
            )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._chat_template is None:
            raise ValueError("Chat template is not set. Please set the chat template before generating chat.")

        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set the tokenizer before generating chat.")
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"
        return (
            chat_input,  # context
            self.dataset[idx]["messages"],  # full message history
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]

# HuggingFaceH4/ultrachat_200k
class AugmentedCoherenceDataset(Dataset):
    def __init__(self, dataset_id: str, split_id: str, max_input_len):
        datass = load_dataset(dataset_id,split=split_id)

        self.dataset = self.process_data(datass, max_input_len)
        self._chat_template = None
        self._tokenizer = None

    def set_chat_template_params(self, template_path: str, tokenizer: AutoTokenizer):
        self._chat_template = jinja2.Template(open(template_path).read())
        self._tokenizer = tokenizer

    def process_data(self, datass, max_input_len):
        """
        Convert dataset to a format that can be used downstream.
        """

        converted_dataset = []

        for entry in datass:
            system_prompt = f'''
            You are roleplaying a character with the name {entry['name']}.
            Your categories and topics of specialty are : {' '.join(entry['categories'])}.
            Your personality can be described with the following : {','.join(entry['personalities'])}.
            Your full character description:
            {entry['description']}
            '''
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            for message in entry['conversation']:
                role = "assistant" if message['role'] == 'character' else 'user'
                messages.append(
                {"role": role, "content": message['content']},
                )
            if messages[-1]["role"] == "assistant":
                del messages[-1]

            converted_dataset.append(
                {
                    "messages": messages,
                }
            )

        return converted_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._chat_template is None:
            raise ValueError("Chat template is not set. Please set the chat template before generating chat.")

        if self._tokenizer is None:
            raise ValueError("Tokenizer is not set. Please set the tokenizer before generating chat.")
        chat_input = self._chat_template.render(
            bos_token=self._tokenizer.bos_token,
            eos_token=self._tokenizer.eos_token,
            messages=self.dataset[idx]["messages"],
            include_beginning_of_conversation=True,
            add_generation_prompt=True,
        )  # shouldn't end with eos token

        if chat_input.endswith(self._tokenizer.eos_token):
            chat_input = chat_input[: -len(self._tokenizer.eos_token)]

        if not chat_input.startswith(self._tokenizer.bos_token):
            chat_input = f"{self._tokenizer.bos_token}{chat_input}"
        return (
            chat_input,  # context
            self.dataset[idx]["messages"],  # full message history
        )

    def sample_dataset(self, n: int):
        # get indices of the dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        return [self[i] for i in indices]