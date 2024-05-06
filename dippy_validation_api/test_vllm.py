from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('Manavshah/llama-test')

tokens = tokenizer(
    [
        tokenizer.bos_token + 'What is the Capital of France?\n',
        tokenizer.bos_token + 'Capital of France:',
    ],
    add_special_tokens=False
)
replies = tokenizer(
    [
        'The capital of France is Paris.' + tokenizer.eos_token,
        'Paris.' + tokenizer.eos_token,
    ],
    add_special_tokens=False
)

prompt_input_ids = tokens['input_ids']
reply_input_ids = replies['input_ids']

full_context = []
for i in range(len(prompt_input_ids)):
    full_context.append(prompt_input_ids[i] + reply_input_ids[i])

# prompts = [
#     'What is the Capital of France?\n',
#     'Capital of France:',
# ]

# replies = [
#     'The capital of France is Paris.',
#     'Paris.',
# ]

# full_context = []
# for i in range(len(prompts)):
#     full_context.append(prompts[i] + replies[i])


llm = LLM('Manavshah/llama-test', max_context_len_to_capture=4096)


llm.llm_engine.model_config.max_logprobs = 100

# Test the model
sampling_params = SamplingParams(
    max_tokens=1,
    prompt_logprobs=100,
)

x = llm.generate(
    prompt_token_ids=full_context, 
    sampling_params=sampling_params,
)

# prompt_log_prob_mappings = {}
# for i in range(len(prompt_input_ids[0]), len(reply_input_ids[0])):
#     prompt_log_prob_mappings[i] = x[0].prompt_logprobs[i]

log_probs = x[0].prompt_logprobs

for i, probs in enumerate(log_probs[len(prompt_input_ids[0]):]):
    candidates = {}
    for token, token_obj in probs.items():
        candidates[token_obj.rank] = token_obj.decoded_token
    
    print(tokenizer.decode(reply_input_ids[0][i]), '->', [candidates[i] for i in range(1, 6)])



