import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-4o-mini"
client = OpenAI(api_key=api_key)

def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def generate_summary(url, temperature=0.1, max_tokens=400):
    article_text = get_article_text(url)
    
    system_context = "You are a useful assistant. Formulate a concise summary of the submitted technical article."
    prompt = f"Article: {article_text}\n\nSummary:"
    
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    url = input("Enter the URL of the article: ")
    temperature = float(input("Enter temperature (0.0 to 1.0): "))
    max_tokens = int(input("Enter max tokens (100 to 1000): "))
    
    summary = generate_summary(url, temperature, max_tokens)
    print("\nGenerated Summary:")
    print(summary)