from openai import OpenAI
import json
import requests
import os

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
)
model = "gpt-4o-mini"

functions = [
    {
        "name": "search_courses",
        "description": "Returns a list of training courses from the Microsoft catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "User role"},
                "product": {"type": "string", "description": "Microsoft product"},
                "level": {"type": "string", "description": "Experience level"},
                "max_duration": {"type": "integer", "description": "Max course duration"},
                "min_rating": {"type": "number", "description": "Minimum average rating"}
            },
            "required": ["product"]
        }
    }
]


messages = [
    {
        "role": "system",
        "content": "Always use and describe results returned by functions. Include title, duration, rating, and difficulty level in your response."
    },
    {
        "role": "user",
        "content": "Find azure-functions courses for beginner developers with a minimum duration of 40 minutes and minimum rating of 4.3."
    }
]

response = client.chat.completions.create(
    model=model,    
    messages=messages,
    functions=functions,
    function_call="auto"
)

response_message = response.choices[0].message

if response_message.function_call:
    function_name = response_message.function_call.name
    arguments = json.loads(response_message.function_call.arguments)

    def search_courses(role=None, product=None, level=None, max_duration=None, min_rating=None):
        url = "https://learn.microsoft.com/api/catalog/"
        params = {}
        if role: params["role"] = role
        if product: params["product"] = product
        if level: params["level"] = level

        response = requests.get(url, params=params)
        data = response.json()
        modules = data.get("modules", []) or data.get("courses", [])

        filtered = []
        for m in modules:
            duration = m.get("duration_in_minutes", 0)
            rating = m.get("rating", {}).get("average", 0)

            if max_duration and duration > max_duration:
                continue
            if "min_duration" in arguments:
                min_duration = arguments["min_duration"]
                if duration < min_duration:
                    continue
            if min_rating and rating < min_rating:
                continue

            filtered.append({
                "title": m.get("title"),
                "url": m.get("url"),
                "summary": m.get("summary"),
                "duration": duration,
                "rating": rating,
                "level": m.get("levels", ["unknown"])[0],
                "type": m.get("type"),
                "locale": m.get("locale")
            })

        return json.dumps(filtered[:5], ensure_ascii=False)

    result = search_courses(**arguments)
    print("Вміст result:")
    print(result)

    messages.append(response_message)
    messages.append({
        "role": "function",
        "name": function_name,
        "content": result
    })

    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    print("📥 Остаточна відповідь:")
    print(final_response.choices[0].message.content)
