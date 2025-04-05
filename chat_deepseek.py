from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
  {
    'role': 'user',
    'content': 'How are you?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)