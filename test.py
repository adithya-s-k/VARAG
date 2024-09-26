from varag.vlms import OpenAI , Groq, Mistral
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

vlm1 = OpenAI()
vlm2 = Groq()
vlm3 = Mistral()

# Single image using __call__ with image path
# response = vlm1("test.png", "What's in this image?")
# print(response)

# print("\n-----------------------------------\n")

response = vlm2("test_images/test.png", "What's in this image?")
print(response)

print("\n-----------------------------------\n")

response = vlm3("test_images/test.png", "What's in this image?")
print(response)

# # Single image using response method with Image object
# image = Image.open("test2.png")
# response = vlm.response("Describe this image", image)
# print(response)

# # Multiple images with max_tokens
# images = ["test.png", "test2.png", Image.open("test.png")]
# response = vlm.response("Compare these images", images, max_tokens=500)
# print(response)