import os
os.system('pip install accelerate')
os.system('pip install ipex-llm[all]')
os.system('pip install modelscope -U')
os.system('pip install --upgrade transformers==4.37.0')
from threading import Thread
from typing import Iterator

import gradio as gr
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import  TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 128
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))


model_id = "qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer.use_default_system_prompt = False
from ipex_llm import optimize_model
model = optimize_model(model)


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer([input_ids],return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids.input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    ) 
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    #dictionary update sequence element #0 has length 19; 2 is required

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

    #outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(outputs)
    #yield outputs
# model_options = ["qwen/Qwen1.5-1.8B-Chat", "qwen/Qwen1.5-1.8B-Chat"]
# model_dropdown = gr.Dropdown(label="Model", choices=model_options)
# model_textbox = gr.Textbox(label="Model ID")

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["你好！你是谁？"],
        ["请简单介绍一下大语言模型?"],
        ["请讲一个小人物成功的故事."],
        ["浙江的省会在哪里?"],
        ["写一篇100字的文章，题目是'人工智能开源的优势'"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("""<center><font size=8>ipex-llm-test👾</center>""")
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

