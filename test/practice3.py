from openai import OpenAI
import os

def generate_text(temperature, max_tokens, scenario, iot_specifications):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4o-mini"
        client = OpenAI(
            api_key=api_key,
        )  
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in optimizing IoT network protocols, analyzing available protocols, and providing the best solution based on resource constraints.",
            },
            {
                "role": "user",
                "content": f"Scenario: {scenario}\nIoT System Specifications: {iot_specifications}",
            },
        ]
        functions = [
    {
        "name": "search_courses",
        "description": "Returns a list of training courses from the Microsoft catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "User role (e.g., developer, student)"
                },
                "subject": {
                    "type": "string",
                    "description": "Covered subject (e.g., Azure, Power BI, etc.)"
                },
                "level": {
                    "type": "string",
                    "description": "User experience level (e.g., beginner)"
                },
                "max_duration": {
                    "type": "integer",
                    "description": "Maximum duration of the course in minutes"
                },
                "min_rating": {
                    "type": "number",
                    "description": "Minimum average rating of the course"
                }
            },
            "required": ["role", "subject"]
        }
    }
]

        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            functions=functions,
    function_call="auto"
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None


def test_application():
    scenario = input("Enter the scenario for the network protocol: ")
    iot_specifications = input("Enter the IoT system specifications (e.g., energy, memory constraints): ")
    temperature = float(input("Enter temperature (0.0 - 1.0): "))
    max_tokens = int(input("Enter max tokens: "))
    
    generated_text = generate_text(temperature, max_tokens, scenario, iot_specifications)
    
    if generated_text:
        print(f"Generated Text: {generated_text}")
    else:
        print("Failed to generate text.")

def main():
    print("IoT Network Protocol Optimization and Analysis Tool")
    
    test_application()

if __name__ == "__main__":
    main()