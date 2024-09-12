import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline
from PIL import Image

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def respond(
    message,
    history: list[tuple[str, str]],
    system_message="You are a friendly Chatbot.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    use_local_model=False,
):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    # Initialize history if it's None
    if history is None:
        history = []

    if use_local_model:
        # local inference 
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for output in pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            token = output['generated_text'][-1]['content']
            response += token
            yield history + [(message, response)]  # Yield history + new response

    else:
        # API-based inference 
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            if stop_inference:
                response = "Inference cancelled."
                break
            token = message_chunk.choices[0].delta.content
            response += token
            yield history + [(message, response)]  # Yield history + new response


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


def cancel_inference():
    global stop_inference
    stop_inference = True

# Custom CSS for a fancy look

custom_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
h1 {
    text-align: center;
    font-size: 40px;
    color: #00008a
}
h3 {
    text-align: center;
    font-size: 20px;
    color: #800020
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
    font-family: 'Robotica';
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #800000;
}
"""


#Max_tokens min/max values, step, randomize

# Define the interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>🤖 Adina and Jai's Chatbot 🤖</h1>")
    gr.Markdown("<h3>Interact with the AI chatbot using customizable settings below.</h3>")

    with gr.Row():
        gr.Image("mlops.png", label="MLOps", type="filepath")
        with gr.Column(scale=4):
            system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True)
            clear_button = gr.ClearButton(system_message)
        gr.Image("WPI.png", label="WPI Logo", type="filepath")

    with gr.Row():
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)
        dup_button = gr.DuplicateButton()

    with gr.Column(scale=4):
        max_tokens = gr.Slider(minimum=1, maximum=3000, value=500, step=50, label="Max new tokens", randomize = True)
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

    chat_history = gr.Chatbot(label="Chat")

    chat_history.like(vote, None, None)

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...", max_lines = 40)

    clear_button_2 = gr.ClearButton(user_input)

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    # Adjusted to ensure history is maintained and passed correctly
    user_input.submit(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)

    cancel_button.click(cancel_inference)
    
    #ClearButton.click(clearbutton)

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=['WPI.png','mlops.png'])  # Remove share=True because it's not supported on HF Spaces
