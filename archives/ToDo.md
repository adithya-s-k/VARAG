## Dataset Preparation

Dataset specification : Image , Caption , Features , Question , Answers
Task Description : Create a pipeline given a PDF each page into a image then do feature extraction using something like [Surya](https://github.com/VikParuchuri/surya) or [marker](https://github.com/VikParuchuri/marker) after that pass the image though [GPT-4 Vision](https://platform.openai.com/docs/guides/vision) with the relavent prompts

- [] Initilise Notebook
- [] Install relavent libraries and repos like Surya and marker
- [] Convert PDF pages to images
- [] Feature extraction
- [] Design a well crafted Prompt for Vision API
- [] Iterate over all the images and corresponding features to generate quesiton answer pairs
- [] Sort the data and push it to Huggingface Hub for easy collboration


## Finetuning 

Task Decription : After the dataset creation is complete(10k - 100k) Data points formate it for instruction finetuning. Test out the following model for finetuning 
- [IDEFICS](https://huggingface.co/docs/transformers/model_doc/idefics)
- [LLaVA](https://llava-vl.github.io/)
- [Moondream](https://huggingface.co/vikhyatk)

- [] Split dataset to train , eval , split
- [] Fromate the dataset
- [] Finalise the finetunign scripts with optimisations
- [] Set up GPU
- [] Finetune with different Lora Configurations
- [] Merge the finetuned adapaters
- [] Set up inference/ generation scripts
- [] Push the model to huggingface hub


## Evaluation 

Task Description : Evaluate the model with different benchmarks(ST-VQA, OCR-VQA, TextVQA, and DocVQA) and show imporvment in performance

- [] Formate evaluation dataset
- [] Finalise Evaluation scripts
- [] Generate Visual Representation of the eval results


## Visual Chat Application


Task Description: Create a visual chat interface integrated with PDFs. Instead of solely relying on text within the PDF, building a chat system that considers both textual and visual inputs would be more useful. Utilize CLIP for image encoding and select a model from the Massive Text Embedding Benchmark (MTEB) leaderboard (accessible at [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)) for textual embedding. Employ a vector database to store both image and text embeddings. During the chat, implement a simple RAG (Retrieval Augmented Generation) pipeline to retrieve relevant text and images from the database. Additionally, incorporate in-context learning for an enhanced user experience.

- [] Create the base application using Gradio
- [] Choose and Set up a vector database
- [] add Upload pdf feature 
- [] Extract everythign from pdf and store in database
- [] Set up the chat interface
- [] Set up inference for LLM and VLM models
- [] Make it google colab compatible
