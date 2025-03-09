import openai

openai.api_key = 'YOUR_API_KEY'

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="こんにちは、元気ですか？",
    max_tokens=50
)

print(response.choices[0].text.strip())
