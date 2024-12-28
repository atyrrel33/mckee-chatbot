from fastapi import FastAPI, File, UploadFile
import pdfplumber

app = FastAPI()

@app.post("/analyze-script/")
async def analyze_script(file: UploadFile = File(...)):
    # Extract text from the uploaded PDF
    with pdfplumber.open(file.file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)

    # Prepare the prompt for the LLM
    prompt = f"Analyze this screenplay based on Robert McKee's principles of story structure, character, and theme:\n\n{text}"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate the analysis
    outputs = model.generate(**inputs, max_length=512)
    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the analysis
    return {"analysis": analysis}

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)