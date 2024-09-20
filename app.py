import os
from groq import Groq
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import gradio as gr
from PIL import Image
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Initialize the Groq client
client = Groq(api_key=api_key)

print("Groq client initialized")

def fetch_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from p, h1, h2, h3 tags
    text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3'])])
    return text[:4000]  # Limit to 4000 characters to avoid token limits

def extract_important_points(text):
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # Use an appropriate Groq model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts important points from text."},
            {"role": "user", "content": f"Extract 5-7 important points from the following text, separated by commas:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content

def create_diagram_from_points(points):
    # Split the points into a list
    point_list = [point.strip() for point in points.split(',')]
    
    # Create a simple flowchart
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(point_list))
    ax.axis('off')

    for i, point in enumerate(point_list):
        y = len(point_list) - i - 0.5
        ax.text(0.5, y, point, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'), wrap=True)
        if i < len(point_list) - 1:
            ax.annotate('', xy=(0.5, y-0.5), xytext=(0.5, y-0.1), arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    
    # Save the plot to a temporary file
    temp_file = 'temp_diagram.png'
    plt.savefig(temp_file, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Open the saved image with PIL
    img = Image.open(temp_file)
    
    # Create a copy of the image in memory
    img_copy = img.copy()
    
    # Close the original image
    img.close()
    
    # Remove the temporary file
    os.remove(temp_file)
    
    return img_copy

def generate_diagram_from_url(url):
    webpage_content = fetch_webpage_content(url)
    important_points = extract_important_points(webpage_content)
    diagram = create_diagram_from_points(important_points)
    return important_points, diagram

# Create Gradio interface
iface = gr.Interface(
    fn=generate_diagram_from_url,
    inputs=gr.Textbox(lines=1, placeholder="Enter a website URL..."),
    outputs=[
        gr.Textbox(label="Extracted Important Points"),
        gr.Image(type="pil", label="Generated Diagram")
    ],
    title="Website Content Diagram Generator",
    description="Enter a URL to extract important points and generate a diagram."
)

# Launch the interface
iface.launch()
