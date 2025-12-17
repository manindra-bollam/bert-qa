## Extractive Question Answering System using BERT

### Overview

This project implements an **end to end Extractive Question Answering (QA) system** using a **pre trained BERT model** fine tuned on the **HotpotQA dataset**. The system takes a **natural language question** and a **context passage** as input and predicts the **most relevant answer span** directly from the given context. In addition to model training and evaluation, the project includes a **web based application** that allows users to interact with the fine tuned model through a simple user interface.


### Motivation

Question Answering is a fundamental Natural Language Processing (NLP) task with applications in **search engines, chatbots, digital assistants, and knowledge retrieval systems**. Traditional keyword based approaches often fail to capture contextual meaning. Transformer based models like **BERT (Bidirectional Encoder Representations from Transformers)** address this limitation by learning deep contextual representations of text.
This project explores how such pre trained models can be adapted to complex QA datasets such as **HotpotQA**, which requires reasoning across multiple sentences.


### Dataset

The project uses the **HotpotQA (distractor setting)** dataset, which contains Wikipedia based question answer pairs designed to evaluate **multi hop reasoning**.
To ensure compatibility with extractive QA:

* Context paragraphs are merged into a single passage.
* Samples with **yes/no answers** are filtered out.
* Only samples where the **answer text appears verbatim in the context** are retained.
* This preprocessing ensures reliable **answer span supervision** during training.


### Methodology

#### 1. Data Preprocessing

* Questions and merged context passages are tokenized using **BERT’s WordPiece tokenizer**.
* Input format follows the standard BERT QA structure:
  `[CLS] Question [SEP] Context [SEP]`
* Character level answer positions are aligned with token offsets to generate:

  * `start_positions`
  * `end_positions`
* Exploratory Data Analysis (EDA) is performed to study:

  * Question types
  * Answer length distributions


#### 2. Model Fine-Tuning

* A **pre trained BERT model** (`bert-base-uncased`) is fine tuned for extractive QA.
* The model learns to predict the **start and end token indices** of the answer span.
* Training is performed on a filtered subset of HotpotQA using:

  * AdamW optimizer
  * Cross entropy loss on start and end logits
* Hyperparameters such as learning rate, batch size, and number of epochs are tuned for stability.


#### 3. Inference Strategy

During inference:

* The model outputs start and end logits for each token.
* The final answer span is selected by **maximizing the joint score of start and end logits** under a maximum span length constraint.
* If no confident span is found, the system safely returns a fallback message.


### Web Application

A **FastAPI-based web application** is developed to serve the fine tuned model.

#### Backend:

* Loads the fine tuned BERT model and tokenizer.
* Exposes a `/predict` API endpoint that accepts JSON input:

  ```json
  {
    "context": "...",
    "question": "..."
  }
  ```
* Returns the predicted answer as a JSON response.

#### Frontend:

* A simple HTML interface allows users to:

  * Enter context text
  * Enter a question
  * View the predicted answer dynamically

This demonstrates a complete **model-to-deployment pipeline**.


### Results and Observations

* The system successfully extracts answers for fact based questions where the answer appears explicitly in the context.
* Due to the complexity of HotpotQA and limited fine tuning data, performance on multi hop and abstract reasoning questions is limited.
* These results align with known challenges in span based QA on multi hop datasets.


### Limitations

* HotpotQA contains many questions that are not well suited for simple extractive QA.
* Long contexts may be truncated due to BERT’s maximum input length.
* The model does not explicitly model reasoning chains or supporting facts.


### Future Enhancements

* Integrating a **retrieval component** to select relevant paragraphs before QA.
* Using **long context transformer architectures** such as Longformer.
* Incorporating **knowledge graphs** for improved multi hop reasoning.
* Extending the system for **cross lingual QA**.
* Improving robustness through adversarial training and data augmentation.


### Conclusion

This project demonstrates how a pre trained transformer model can be adapted for extractive question answering and deployed as an interactive web application. It highlights both the strengths of BERT based QA systems and the challenges posed by complex multi hop datasets like HotpotQA. The project provides a solid foundation for further research into advanced QA architectures and reasoning based NLP systems.

