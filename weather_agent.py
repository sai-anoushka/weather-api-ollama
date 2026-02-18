import requests
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Step 1: Define the weather function using Open-Meteo (no API key needed)
def get_weather(city: str) -> str:
    """Get the current weather for a given city using Open-Meteo."""
    # First, get lat/long from geocoding API
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
    geo_response = requests.get(geo_url, params=geo_params)
    
    if geo_response.status_code == 200 and geo_response.json().get("results"):
        result = geo_response.json()["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        country = result.get("country", "Unknown")
        
        # Now fetch weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "weather_code"],  # Basic current weather
            "temperature_unit": "celsius",
            "timezone": "auto"
        }
        weather_response = requests.get(weather_url, params=weather_params)
        
        if weather_response.status_code == 200:
            data = weather_response.json()["current"]
            temp = data["temperature_2m"]
            code = data["weather_code"]
            # Simple weather code mapping (from Open-Meteo docs)
            descriptions = {
                0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
                55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
                71: "slight snow", 73: "moderate snow", 75: "heavy snow", 80: "slight rain showers",
                81: "moderate rain showers", 82: "violent rain showers", 95: "thunderstorm",
                # Add more if needed
            }
            desc = descriptions.get(code, "unknown conditions")
            return f"In {city}, {country}, it's currently {desc} with a temperature of {temp}°C."
    return "Sorry, couldn't fetch weather for that city. Try a different name?"

# Step 2: Set up the Ollama model
llm = ChatOllama(model="llama3.2:1b", temperature=0.2)  # Slight temp increase for natural flow

# Step 3: Define a prompt for natural, human-like responses
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a friendly, casual assistant. Respond like a human in normal conversation—be helpful, chatty, and natural. For weather questions, suggest the tool in JSON ONLY if needed. Do NOT add extra text to JSON suggestions.
    
ONLY suggest the tool for weather-related queries. For everything else (like greetings), chat normally.
    
To suggest the tool: Output PURE JSON like: {"tool": "get_weather", "args": {"city": "CityName"}}.
    
Examples:
User: hi
Response: Hey there! What's up?

User: What's the weather in Paris?
Response: {"tool": "get_weather", "args": {"city": "Paris"}}

When you get a tool result (like "Tool result: In Paris, France, it's clear sky with 15°C."), summarize it casually WITHOUT inventing details. Use ONLY the provided info.
Example response: Sure, let me check... Oh, in Paris, it's clear skies at 15°C right now. Anything else?"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Step 4: Function to handle tool suggestions in a loop
def agent_loop(user_input: str):
    messages = [HumanMessage(content=user_input)]
    max_loops = 3  # Prevent infinite loops
    loop_count = 0
    
    while loop_count < max_loops:
        loop_count += 1
        # Invoke the model
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        print("Debug: Raw AI output:", response.content)  # For debugging
        messages.append(AIMessage(content=response.content))
        
        # Check for tool suggestion
        try:
            tool_call = json.loads(response.content.strip())
            if isinstance(tool_call, dict) and tool_call.get("tool") == "get_weather":
                city = tool_call["args"].get("city")
                if city:
                    tool_result = get_weather(city)
                    print("Debug: Real tool result:", tool_result)  # Confirm API call
                    messages.append(HumanMessage(content=f"Tool result: {tool_result}"))
                    continue  # Let model summarize naturally
        except json.JSONDecodeError:
            pass  # Normal response, not tool
        
        # Return the final natural response
        return response.content

# Step 5: Run the agent in a chat loop
print("Chat with the agent (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    output = agent_loop(user_input)
    print("Agent:", output)