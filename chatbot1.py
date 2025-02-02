def chatbot_response(user_message):
    import json
    import google.generativeai as genai
    from fuzzywuzzy import process
    
    # Load data from JSON file
    def load_data():
        with open("collegedata.json", "r") as file:
            return json.load(file)

    data = load_data()

# Set up Gemini API
    GENAI_API_KEY = "AIzaSyAJboTnIeCT0SMkylUotr3ipdwiVjbjl54"  # Replace with your API key
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")

# Function to find the best match from JSON
    def find_best_match(user_message):
        questions = list(data.keys())
        best_match, score = process.extractOne(user_message, questions)
        return data[best_match] if score > 60 else None

# Function to get response from Gemini API
    def get_gemini_response(user_input):
        response = model.generate_content(user_input)
        return response.text if response else "I'm unable to process this request."

# Chat loop
    print("MECCo: Ask me anything ! Type 'exit' to quit.")
    while True:
        if user_message.lower() == "exit":
            print("MECCo: Goodbye!")
            break

    # First check JSON, if no match, use Gemini
        response = find_best_match(user_message)
        if response:
            print("MECCo:", response)
            break
        else:
            print("MECCo(Gemini):", get_gemini_response(user_message)) 
            break 
message=input('enter your question')
chatbot_response(message)