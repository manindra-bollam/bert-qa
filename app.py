from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

# Initialize FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Load fine-tuned model and tokenizer
MODEL_PATH = "./bert-qa-model"

# MODEL_PATH = "bert-large-uncased-whole-word-masking-finetuned-squad"


tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
model.eval()

# Request schema
class QARequest(BaseModel):
    context: str
    question: str

@app.post("/predict")
def predict_answer(data: QARequest):
    inputs = tokenizer(
        data.question,
        data.context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    max_score = float("-inf")
    best_start, best_end = 0, 0
    max_answer_length = 30

    for start_idx in range(len(start_logits)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > max_score:
                best_start = start_idx
                best_end = end_idx
                max_score = score

    answer_tokens = inputs["input_ids"][0][best_start:best_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    if answer == "":
        return {"answer": "No confident answer found"}

    return {"answer": answer}



