# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json
import time
import datasets
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
boto_session = boto3.Session(profile_name="Nathan")
client = boto_session.client("bedrock-runtime", region_name="us-west-2")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"


passages = datasets.load_dataset("Malikeh1375/medical-question-answering-datasets", 'all-processed',split="train",trust_remote_code=True)['output']

def create_prompt(dataset, i, prefix) :
    text = dataset[i]
    return prefix + text

prefix = 'Find an question that could have been asked by a patient using a chatbot, to which a doctor answered the following text (Do not include any presentation or sentence for introduction, simply write the most plausible question as if you were the patient.): '

# Function to invoke the model.
def invoke_model(prompt, model_id):
    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)
        # Decode the response body.
        model_response = json.loads(response["body"].read())
        # Extract and return the response text.
        response_text = model_response["content"][0]["text"]
        return response_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None

# Loop to send 100 separate requests.
responses = []
for i in range(5):
    print(f"Sending request {i + 1}")
    prompt = create_prompt(passages, i, prefix)
    response_text = invoke_model(prompt, model_id)
    if response_text:
        responses.append(response_text)
    # Optional: sleep to avoid hitting rate limits
    #time.sleep(0.1)  # Adjust sleep time as needed

# Optionally, save responses to a file
with open("Results/responses.txt", "w") as f:
    for idx, response in enumerate(responses, 1):
        f.write(f"Response {idx}: {response}\n")