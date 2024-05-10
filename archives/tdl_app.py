from PyPDF2 import PdfReader
from pdf2jpg import pdf2jpg
import gradio as gr
import shutil
import os
import chromadb
from chromadb.utils import embedding_functions



chroma_client = chromadb.Client()


def upload_file(file):
    # saving pdf
    # path = r"C:\Users\bhara\Desktop\hhh\\" + os.path.basename(files[0])
    # shutil.copyfile(files[0].name, path)

    global chroma_client
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(name="pdf_embds", embedding_function=ef)


    # print text in pdf
    reader = PdfReader(file)
    print(len(reader.pages)) 
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        page_content = page.extract_text()  
        print(page_content)


        # saving to chromadb
        collection.add(
        # embeddings=page_embedding,
        documents=[page_content],
        ids=[f"pg{i}"]
        )

        print(collection.get(
            include=["embeddings"],
            ids=[f"pg{i}"])
        )
        print("============================================================")



    print(collection.count())

    # saving pdf as imgs
    inputpath = file
    outputpath = r""
    result = pdf2jpg.convert_pdf2jpg(inputpath,outputpath, pages="ALL")






with gr.Blocks() as demo:
        # file_output = gr.File()
        upload_button = gr.UploadButton("Click to Upload a File", file_count="single")
        upload_button.upload(upload_file, upload_button)


demo.launch()
