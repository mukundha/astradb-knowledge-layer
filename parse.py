import pdf2image
import PIL
from google import genai
from google.genai import types  
import io
import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
import sys 

load_dotenv()  # Load environment variables from .env file

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)


class ChunkMetadata(BaseModel):
    entities: List[str]
    topics: List[str]


class Chunk(BaseModel):
    chunk: str
    metadata: ChunkMetadata

def pdf_to_images(pdf_path: str) -> List[PIL.Image.Image]:
    """
    Converts a PDF file to a list of PIL images.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A list of PIL images, one for each page of the PDF.
    """
    try:
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def image_to_markdown(image: PIL.Image.Image) -> list[Chunk]:
    """
    Uses Gemini to OCR an image and convert it to Markdown, formatting tables as HTML.
    This version uses Gemini to OCR and chunk the text directly and ensures valid JSON output.

    Args:
        image: The PIL image to OCR.

    Returns:
        The JSON string representation of the chunks and metadata extracted by Gemini.
    """

    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        prompt = """
            You are an expert at extracting information from images, especially PDFs. 
            Convert the image to markdown, format tables as HTML. Preserve all information.
            
            Do not surround your output with triple backticks or any other formatting markers outside of the JSON itself.

            Chunk the document into sections of roughly 250 - 1000 words. Our goal is 
            to identify parts of the page with the same semantic theme. These chunks will 
            be embedded and used in a RAG pipeline. 

            Return the response as a **single, valid JSON array** of objects, with chunks and metadata containing key entities and topics referenced by the chunk.  Ensure the JSON is well-formed and valid. 
            Do not surround your output with triple backticks or any other formatting markers outside of the JSON itself.

            For example:
            
            [
              {
                "chunk": "chunk1",
                "metadata": {
                  "entities": ["entity1", "entity2"],
                  "topics": ["topic1", "topic2"]
                }
              },
              {
                "chunk": "chunk2",
                "metadata": {
                  "entities": ["entity3", "entity4"],
                  "topics": ["topic3", "topic4"]
                }
              }
            ]
            
        """
        config= {
            'response_mime_type': 'application/json',
            'response_schema': list[Chunk], 
        }
        response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents= [prompt, types.Part.from_bytes(data=img_data, mime_type="image/png" )],
                config= config
                )        
        json = response.parsed        
        return json 
    except Exception as e:
        print(f"Error OCRing image: {e}")
        return '[]'  # Return an empty JSON array


def process_pdf(pdf_path: str) -> str:
    """
    Processes a PDF file: converts it to images, OCRs each image using Gemini to also chunk and extract metadata,
    and returns a JSON string containing the chunks and metadata.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A JSON string representing the processed chunks and metadata.
    """
    images = pdf_to_images(pdf_path)
    if not images:
        return json.dumps({"error": "Could not process PDF."})

    all_chunks = []
    for image in images:
        json_page = image_to_markdown(image)
        all_chunks.extend(json_page)
    return all_chunks


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python parse.py <pdf_file_path>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    if not os.path.exists(pdf_file_path):
        print(f"File '{pdf_file_path}' does not exist. Please provide a valid PDF file.")
        sys.exit(1)

    processed_data = process_pdf(pdf_file_path)    
    chunks_dict = [chunk.model_dump() for chunk in processed_data] 
    json_string = json.dumps(chunks_dict, indent=2)
    try:
        with open("out.json", "w") as f:
            f.write(json_string)
        print("Data written to out.json")
    except Exception as e:
        print(f"Error writing to file: {e}")