from mlx_lm import load, generate

model, tokenizer = load('mistralai/Mistral-7B-Instruct-v0.1')
response = generate(model, tokenizer, prompt = 'Write a haiku about flowers', verbose = True)
print(response)