import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

with open("knowledge.txt", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable "r" shows that we're gonna open this in read mode
  knowledge_base = file.read()

# Print the text below
print(knowledge_base)

def preprocess_text(text):
    cleaned_text = text.strip()

    chunks = cleaned_text.split("\n")

    cleaned_chunks = []

    for chunk in chunks:
        stripped_chunk = chunk.strip()
        cleaned_chunks.append(stripped_chunk)

    print(cleaned_chunks)
    print(len(cleaned_chunks))

    return cleaned_chunks

cleaned_chunks = preprocess_text(knowledge_base)

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)


  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)


  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks)

def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
        relevant_info = text_chunks[i]
        top_chunks.append(relevant_info)

    # Return the list of most relevant chunks
  return top_chunks

    

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")


def respond(message, history):
    
    rag_info = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    system_message = f"""
    You are KodeBot, a friendly chatbot.

    Use ONLY the information below to answer questions.
    If the answer is not in the information, say:
    "I don't know based on the provided knowledge."

    Keep your answer under 40 words.

    Knowledge:
    {rag_info}
    """    
    messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "What is Kode with Klossy?"},
    {"role": "assistant", "content": "Kode with Klossy is a program that teaches coding and technology skills to students."}
    ]
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=1.5,
        top_p=0.3
    )
    return response['choices'][0]['message']['content'].strip() 


with gr.Blocks() as chatbot:
    gr.Image("BudgetBuddy_Image_Header.png", show_label=False)
    
    gr.ChatInterface(respond,
    title = "Hi, I'm BudgetBuddy! 💵",
    textbox= gr.Textbox(placeholder="Share Your Budget Or Ask Me Anything!"),
    description = "A smart chatbot that combines budgeting and mental wellness to help you spend mindfully, save better, and stress less!",
    examples = ["I get $500 per month, can you make me a budget?", 
                "I keep buying things when I'm stressed",
                "Help me save for a new phone"]
                    )


chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
