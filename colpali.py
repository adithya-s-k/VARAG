# import gradio as gr
# import os
# import base64
# from byaldi import RAGMultiModalModel
# from PIL import Image
# import io
# from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
# import torch
# from threading import Thread
# import time
# import fitz  # PyMuPDF

# # Initialize the RAG model
# RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=0)

# # Initialize the LLM
# ckpt = "nltpt/Llama-3.2-11B-Vision-Instruct"
# CUDA_LAUNCH_BLOCKING = 1
# model = MllamaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to("cuda")
# processor = AutoProcessor.from_pretrained(ckpt)

# def ingest_pdf(file):
#     if file is None:
#         return "No file uploaded."
    
#     # Index the PDF
#     RAG.index(
#         input_path=file.name,
#         index_name="uploaded_pdf",
#         store_collection_with_index=True,
#         overwrite=True
#     )
    
#     return "PDF ingested successfully."

# def search_pdf(query):
#     if not query:
#         return "Please enter a search query.", []
    
#     results = RAG.search(query, k=5)  # Limit to top 5 results
#     if not results:
#         return "No results found.", []
    
#     images = []
#     for result in results:
#         image_base64 = result.base64
#         image_bytes = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_bytes))
#         images.append(image)
    
#     return f"Found {len(images)} results.", images

# def bot_streaming(message, history, pdf_file, max_new_tokens=250):
#     txt = message["text"]
    
#     messages = []
#     images = []

#     # Search PDF using RAG
#     search_status, search_images = search_pdf(txt)
#     if search_images:
#         images.extend(search_images)
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": f"Based on the search query '{txt}', here are some relevant images:"}] + 
#                        [{"type": "image"} for _ in search_images]
#         })

#     # Process conversation history
#     for i, msg in enumerate(history):
#         if isinstance(msg[0], tuple):
#             images_in_turn = [Image.open(img).convert("RGB") for img in msg[0]]
#             messages.append({
#                 "role": "user", 
#                 "content": [{"type": "text", "text": history[i+1][0]}] + [{"type": "image"} for _ in images_in_turn]
#             })
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": history[i+1][1]}]})
#             images.extend(images_in_turn)
#         elif isinstance(history[i-1], tuple) and isinstance(msg[0], str):
#             pass
#         elif isinstance(history[i-1][0], str) and isinstance(msg[0], str):
#             messages.append({"role": "user", "content": [{"type": "text", "text": msg[0]}]})
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": msg[1]}]})

#     # Add current message
#     current_images = []
#     for file in message["files"]:
#         if isinstance(file, str):
#             image = Image.open(file).convert("RGB")
#         else:
#             image = Image.open(file["path"]).convert("RGB")
#         current_images.append(image)
    
#     if current_images:
#         images.extend(current_images)
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": txt}] + [{"type": "image"} for _ in current_images]
#         })
#     else:
#         messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})

#     texts = processor.apply_chat_template(messages, add_generation_prompt=True)

#     inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
    
#     streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)

#     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    
#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()
#     buffer = ""
    
#     for new_text in streamer:
#         buffer += new_text
#         time.sleep(0.01)
#         yield buffer

# demo = gr.Blocks()

# with demo:
#     gr.Markdown("# Multimodal Llama with PDF Support and RAG Search")
#     gr.Markdown("Upload a PDF, ingest it, and start chatting. The system will search the PDF for relevant images based on your queries.")
    
#     with gr.Row():
#         pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
#         ingest_button = gr.Button("Ingest PDF")
    
#     ingest_output = gr.Textbox(label="Ingestion Status")
    
#     ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    
#     chatbot = gr.ChatInterface(
#         fn=bot_streaming,
#         additional_inputs=[
#             pdf_input,
#             gr.Slider(
#                 minimum=10,
#                 maximum=8000,
#                 value=250,
#                 step=10,
#                 label="Maximum number of new tokens to generate",
#             )
#         ],
#         title="",
#         textbox=gr.MultimodalTextbox(),
#         cache_examples=False,
#         description="Chat about the ingested PDF. The system will search for relevant images based on your queries.",
#         stop_btn="Stop Generation",
#         fill_height=True,
#         multimodal=True
#     )

# demo.launch(debug=True)
# import gradio as gr
# import os
# import base64
# from byaldi import RAGMultiModalModel
# from PIL import Image
# import io
# from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
# import torch
# from threading import Thread
# import time

# # Initialize the RAG model
# RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=0)

# # Initialize the LLM
# ckpt = "nltpt/Llama-3.2-11B-Vision-Instruct"
# CUDA_LAUNCH_BLOCKING = 1
# model = MllamaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to("cuda")
# processor = AutoProcessor.from_pretrained(ckpt)

# def ingest_pdf(file):
#     if file is None:
#         return "No file uploaded."
    
#     # Index the PDF
#     RAG.index(
#         input_path=file.name,
#         index_name="uploaded_pdf",
#         store_collection_with_index=True,
#         overwrite=True
#     )
    
#     return "PDF ingested successfully."

# def search_pdf(query):
#     if not query:
#         return "Please enter a search query.", []
    
#     results = RAG.search(query, k=5)  # Limit to top 5 results
#     if not results:
#         return "No results found.", []
    
#     images = []
#     for result in results:
#         image_base64 = result.base64
#         image_bytes = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_bytes))
#         images.append(image)
    
#     return f"Found {len(images)} results.", images

# def bot_streaming(message, history, pdf_file, max_new_tokens=250):
#     txt = message["text"]
    
#     messages = []
#     images = []

#     # Search PDF using RAG
#     search_status, search_images = search_pdf(txt)
    
#     # Prepare the response with retrieved images
#     response = [(txt, None)]
#     if search_images:
#         images.extend(search_images)
#         image_message = f"Based on the search query '{txt}', here are some relevant images:"
#         response.append((image_message, search_images))
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": image_message}] + 
#                        [{"type": "image"} for _ in search_images]
#         })

#     # Process conversation history
#     for i, msg in enumerate(history):
#         if isinstance(msg[0], tuple):
#             images_in_turn = [Image.open(img).convert("RGB") for img in msg[0]]
#             messages.append({
#                 "role": "user", 
#                 "content": [{"type": "text", "text": history[i+1][0]}] + [{"type": "image"} for _ in images_in_turn]
#             })
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": history[i+1][1]}]})
#             images.extend(images_in_turn)
#         elif isinstance(history[i-1], tuple) and isinstance(msg[0], str):
#             pass
#         elif isinstance(history[i-1][0], str) and isinstance(msg[0], str):
#             messages.append({"role": "user", "content": [{"type": "text", "text": msg[0]}]})
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": msg[1]}]})

#     # Add current message
#     current_images = []
#     for file in message["files"]:
#         if isinstance(file, str):
#             image = Image.open(file).convert("RGB")
#         else:
#             image = Image.open(file["path"]).convert("RGB")
#         current_images.append(image)
    
#     if current_images:
#         images.extend(current_images)
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": txt}] + [{"type": "image"} for _ in current_images]
#         })
#     else:
#         messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})

#     texts = processor.apply_chat_template(messages, add_generation_prompt=True)

#     inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
    
#     streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)

#     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    
#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()
#     buffer = ""
    
#     for new_text in streamer:
#         buffer += new_text
#         time.sleep(0.01)
#         yield response + [(None, buffer)]

# demo = gr.Blocks()

# with demo:
#     gr.Markdown("# Multimodal Llama with PDF Support and RAG Search")
#     gr.Markdown("Upload a PDF, ingest it, and start chatting. The system will search the PDF for relevant images based on your queries.")
    
#     with gr.Row():
#         pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
#         ingest_button = gr.Button("Ingest PDF")
    
#     ingest_output = gr.Textbox(label="Ingestion Status")
    
#     ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    
#     chatbot = gr.Chatbot(
#         [],
#         elem_id="chatbot",
#         avatar_images=(None, "https://picsum.photos/seed/user/200/200"),
#         height=750,
#     )

#     with gr.Row():
#         msg = gr.Textbox(
#             scale=4,
#             show_label=False,
#             placeholder="Enter text and press enter, or upload an image",
#             container=False,
#         )
#         btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

#     with gr.Row():
#         submit = gr.Button("Submit")
#         clear = gr.Button("Clear")

#     max_new_tokens = gr.Slider(
#         minimum=10,
#         maximum=8000,
#         value=250,
#         step=10,
#         label="Maximum number of new tokens to generate",
#     )

#     def user(user_message, history):
#         return "", history + [[user_message, None]]

#     def bot(history, pdf_file, max_new_tokens):
#         if len(history) == 0:
#             return history

#         user_message = history[-1][0]
#         if isinstance(user_message, tuple):
#             user_message = {"text": user_message[0], "files": user_message[1:]}
#         else:
#             user_message = {"text": user_message, "files": []}

#         bot_response = bot_streaming(user_message, history[:-1], pdf_file, max_new_tokens)
#         history[-1][1] = ""

#         for chunk in bot_response:
#             history[-1][1] = chunk[-1][1]
#             if len(chunk) > 1:
#                 history = history + chunk[:-1]
#             yield history

#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, [chatbot, pdf_input, max_new_tokens], chatbot
#     )
#     submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, [chatbot, pdf_input, max_new_tokens], chatbot
#     )
#     btn.upload(lambda x: x.name, inputs=btn, outputs=msg, queue=False)
#     clear.click(lambda: [], outputs=[chatbot], queue=False)

# demo.queue()
# demo.launch()
# import gradio as gr
# import os
# import base64
# from byaldi import RAGMultiModalModel
# from PIL import Image
# import io
# from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
# import torch
# from threading import Thread
# import time
# import tempfile
# import uuid

# # Initialize the RAG model
# RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=0)

# # Initialize the LLM
# ckpt = "nltpt/Llama-3.2-11B-Vision-Instruct"
# CUDA_LAUNCH_BLOCKING = 1
# model = MllamaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to("cuda")
# processor = AutoProcessor.from_pretrained(ckpt)

# # Create a temporary directory to store images
# temp_dir = tempfile.mkdtemp()

# def save_image_temp(image):
#     filename = f"{uuid.uuid4()}.png"
#     filepath = os.path.join(temp_dir, filename)
#     image.save(filepath)
#     return filepath

# def ingest_pdf(file):
#     if file is None:
#         return "No file uploaded."
    
#     # Index the PDF
#     RAG.index(
#         input_path=file.name,
#         index_name="uploaded_pdf",
#         store_collection_with_index=True,
#         overwrite=True
#     )
    
#     return "PDF ingested successfully."

# def search_pdf(query):
#     if not query:
#         return "Please enter a search query.", []
    
#     results = RAG.search(query, k=5)  # Limit to top 5 results
#     if not results:
#         return "No results found.", []
    
#     image_paths = []
#     for result in results:
#         image_base64 = result.base64
#         image_bytes = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_bytes))
#         image_path = save_image_temp(image)
#         image_paths.append(image_path)
    
#     return f"Found {len(image_paths)} results.", image_paths

# def bot_streaming(message, history, pdf_file, max_new_tokens=250):
#     txt = message["text"]
    
#     messages = []
#     images = []

#     # Search PDF using RAG
#     search_status, search_image_paths = search_pdf(txt)
    
#     # Prepare the response with retrieved images
#     response = [(txt, None)]
#     if search_image_paths:
#         images = [Image.open(path) for path in search_image_paths]
#         image_message = f"Based on the search query '{txt}', here are some relevant images:"
#         response.append((image_message, search_image_paths))
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": image_message}] + 
#                        [{"type": "image"} for _ in images]
#         })

#     # Process conversation history
#     for i, msg in enumerate(history):
#         if isinstance(msg[0], tuple):
#             images_in_turn = [Image.open(img).convert("RGB") for img in msg[0]]
#             messages.append({
#                 "role": "user", 
#                 "content": [{"type": "text", "text": history[i+1][0]}] + [{"type": "image"} for _ in images_in_turn]
#             })
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": history[i+1][1]}]})
#             images.extend(images_in_turn)
#         elif isinstance(history[i-1], tuple) and isinstance(msg[0], str):
#             pass
#         elif isinstance(history[i-1][0], str) and isinstance(msg[0], str):
#             messages.append({"role": "user", "content": [{"type": "text", "text": msg[0]}]})
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": msg[1]}]})

#     # Add current message
#     current_images = []
#     for file in message["files"]:
#         if isinstance(file, str):
#             image = Image.open(file).convert("RGB")
#         else:
#             image = Image.open(file["path"]).convert("RGB")
#         current_images.append(image)
    
#     if current_images:
#         images.extend(current_images)
#         messages.append({
#             "role": "user", 
#             "content": [{"type": "text", "text": txt}] + [{"type": "image"} for _ in current_images]
#         })
#     else:
#         messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})

#     texts = processor.apply_chat_template(messages, add_generation_prompt=True)

#     inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
    
#     streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)

#     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    
#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()
#     buffer = ""
    
#     for new_text in streamer:
#         buffer += new_text
#         time.sleep(0.01)
#         yield response + [(None, buffer)]

# demo = gr.Blocks()

# with demo:
#     gr.Markdown("# Multimodal Llama with PDF Support and RAG Search")
#     gr.Markdown("Upload a PDF, ingest it, and start chatting. The system will search the PDF for relevant images based on your queries.")
    
#     with gr.Row():
#         pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
#         ingest_button = gr.Button("Ingest PDF")
    
#     ingest_output = gr.Textbox(label="Ingestion Status")
    
#     ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    
#     chatbot = gr.Chatbot(
#         [],
#         elem_id="chatbot",
#         avatar_images=(None, "https://picsum.photos/seed/user/200/200"),
#         height=750,
#     )

#     with gr.Row():
#         msg = gr.Textbox(
#             scale=4,
#             show_label=False,
#             placeholder="Enter text and press enter, or upload an image",
#             container=False,
#         )
#         btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

#     with gr.Row():
#         submit = gr.Button("Submit")
#         clear = gr.Button("Clear")

#     max_new_tokens = gr.Slider(
#         minimum=10,
#         maximum=8000,
#         value=250,
#         step=10,
#         label="Maximum number of new tokens to generate",
#     )

#     def user(user_message, history):
#         return "", history + [[user_message, None]]

#     def bot(history, pdf_file, max_new_tokens):
#         if len(history) == 0:
#             return history

#         user_message = history[-1][0]
#         if isinstance(user_message, tuple):
#             user_message = {"text": user_message[0], "files": user_message[1:]}
#         else:
#             user_message = {"text": user_message, "files": []}

#         bot_response = bot_streaming(user_message, history[:-1], pdf_file, max_new_tokens)
#         history[-1][1] = ""

#         for chunk in bot_response:
#             new_history = history.copy()
#             new_history[-1] = list(new_history[-1])  # Convert the last item to a list
#             new_history[-1][1] = chunk[-1][1]
#             if len(chunk) > 1:
#                 new_history.extend([list(item) for item in chunk[:-1]])  # Add new items as lists
#             yield new_history


#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, [chatbot, pdf_input, max_new_tokens], chatbot
#     )
#     submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, [chatbot, pdf_input, max_new_tokens], chatbot
#     )
#     btn.upload(lambda x: x.name, inputs=btn, outputs=msg, queue=False)
#     clear.click(lambda: [], outputs=[chatbot], queue=False)

# demo.queue()
# demo.launch(share=True)


import gradio as gr
import os
import base64
from byaldi import RAGMultiModalModel
from PIL import Image
import io
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
import torch
from threading import Thread
import time
import tempfile
import uuid

# Initialize the RAG model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=0)

# Initialize the LLM
ckpt = "nltpt/Llama-3.2-11B-Vision-Instruct"
CUDA_LAUNCH_BLOCKING = 1
model = MllamaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to("cuda")
processor = AutoProcessor.from_pretrained(ckpt)

# Create a temporary directory to store images
temp_dir = tempfile.mkdtemp()

def save_image_temp(image):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(temp_dir, filename)
    image.save(filepath)
    return filepath

def ingest_pdf(file):
    if file is None:
        return "No file uploaded."
    
    # Index the PDF
    RAG.index(
        input_path=file.name,
        index_name="uploaded_pdf",
        store_collection_with_index=True,
        overwrite=True
    )
    
    return "PDF ingested successfully."

def search_pdf(query):
    if not query:
        return "Please enter a search query.", []
    
    results = RAG.search(query, k=5)  # Limit to top 5 results
    if not results:
        return "No results found.", []
    
    image_paths = []
    for result in results:
        image_base64 = result.base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_path = save_image_temp(image)
        image_paths.append(image_path)
    
    return f"Found {len(image_paths)} results.", image_paths

def bot_streaming(message, history, pdf_file, max_new_tokens=250):
    txt = message["text"]
    
    messages = []
    images = []

    # Search PDF using RAG
    search_status, search_image_paths = search_pdf(txt)
    
    # Prepare the response with retrieved images
    if search_image_paths:
        images = [Image.open(path) for path in search_image_paths]
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": txt}] + 
                       [{"type": "image"} for _ in images]
        })

    # Process conversation history
    for i, msg in enumerate(history):
        if isinstance(msg[0], tuple):
            images_in_turn = [Image.open(img).convert("RGB") for img in msg[0]]
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": history[i+1][0]}] + [{"type": "image"} for _ in images_in_turn]
            })
            messages.append({"role": "assistant", "content": [{"type": "text", "text": history[i+1][1]}]})
            images.extend(images_in_turn)
        elif isinstance(history[i-1], tuple) and isinstance(msg[0], str):
            pass
        elif isinstance(history[i-1][0], str) and isinstance(msg[0], str):
            messages.append({"role": "user", "content": [{"type": "text", "text": msg[0]}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": msg[1]}]})

    # Add current message
    current_images = []
    for file in message["files"]:
        if isinstance(file, str):
            image = Image.open(file).convert("RGB")
        else:
            image = Image.open(file["path"]).convert("RGB")
        current_images.append(image)
    
    if current_images:
        images.extend(current_images)
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": txt}] + [{"type": "image"} for _ in current_images]
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": txt}]})

    texts = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(text=texts, images=images, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(processor, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, search_image_paths

demo = gr.Blocks()

with demo:
    gr.Markdown("# Multimodal Llama with PDF Support and RAG Search")
    gr.Markdown("Upload a PDF, ingest it, and start chatting. The system will search the PDF for relevant images based on your queries.")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_button = gr.Button("Ingest PDF")
    
    ingest_output = gr.Textbox(label="Ingestion Status")
    
    ingest_button.click(ingest_pdf, inputs=[pdf_input], outputs=[ingest_output])
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        avatar_images=(None, "https://picsum.photos/seed/user/200/200"),
        height=750,
    )

    image_gallery = gr.Gallery(label="Retrieved Images", show_label=True, elem_id="gallery", columns=5, height=300)

    with gr.Row():
        msg = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    with gr.Row():
        submit = gr.Button("Submit")
        clear = gr.Button("Clear")

    max_new_tokens = gr.Slider(
        minimum=10,
        maximum=8000,
        value=250,
        step=10,
        label="Maximum number of new tokens to generate",
    )

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, pdf_file, max_new_tokens):
        if len(history) == 0:
            return history, []

        user_message = history[-1][0]
        if isinstance(user_message, tuple):
            user_message = {"text": user_message[0], "files": user_message[1:]}
        else:
            user_message = {"text": user_message, "files": []}

        bot_response = bot_streaming(user_message, history[:-1], pdf_file, max_new_tokens)
        history[-1][1] = ""

        for response, image_paths in bot_response:
            new_history = history.copy()
            new_history[-1] = list(new_history[-1])
            new_history[-1][1] = response
            yield new_history, image_paths

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, pdf_input, max_new_tokens], [chatbot, image_gallery]
    )
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, pdf_input, max_new_tokens], [chatbot, image_gallery]
    )
    btn.upload(lambda x: x.name, inputs=btn, outputs=msg, queue=False)
    clear.click(lambda: ([], []), outputs=[chatbot, image_gallery], queue=False)

demo.queue()
demo.launch(share=True)