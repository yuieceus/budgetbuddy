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
    You are BudgetBuddy, a friendly chatbot.

    Use ONLY the information below to answer questions.
    If the answer is not in the information, say:
    "I don't know based on the provided knowledge."

    Keep your answer under 100 words.

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
        max_tokens=300,
        temperature=1.5,
        top_p=0.3
    )
    return response['choices'][0]['message']['content'].strip() 


with gr.Blocks() as chatbot:
    gr.Image("BudgetBuddy_Image_Header1.png", show_label=False)
    
    gr.ChatInterface(respond,
    title = "Hi, I'm BudgetBuddy!💵",
    textbox= gr.Textbox(placeholder="Share Your Budget or Ask Me Anything!"),
    description = "A smart chatbot that combines budgeting and mental wellness to help you spend mindfully, save better, and stress less!",
    examples = ["I get $500 per month, can you make me a budget?", 
                "I keep buying things when I'm stressed. What should I do?",
                "I spent $40 on food and $60 on clothes. Can you analyze it?",
                "How can I save more?"]
                    )


chatbot.launch()


###########################################################
#               EXTRA FEATURES - BudgetBuddy
###########################################################

import random
import datetime

# -------------------------
# Budget Tips
# -------------------------

budget_tips = [
    "Save before you spend.",
    "Track every purchase, even the small ones.",
    "Needs come before wants.",
    "Use the 24-hour rule before buying something expensive.",
    "Create one savings goal each month.",
    "Avoid shopping when you're stressed.",
    "Compare prices before making purchases.",
    "Small daily savings become big yearly savings.",
    "Always keep a small emergency fund.",
    "Review your budget once every week."
]

def random_tip():
    return f"💡 **Today's Budget Tip**\n\n{random.choice(budget_tips)}"


# -------------------------
# Daily Saving Challenge
# -------------------------

saving_challenges = [
    "💰 Save $5 today.",
    "🚫 Buy nothing that isn't necessary today.",
    "🥤 Skip buying one drink and save the money.",
    "🍱 Bring food from home today.",
    "🪙 Put every coin you receive into savings.",
    "📒 Track every purchase you make today.",
    "🛍️ Avoid online shopping for 24 hours.",
    "🚶 Walk instead of paying for transportation if possible.",
    "📚 Read one article about personal finance.",
    "🎯 Move $10 into your savings account."
]

def random_challenge():
    return random.choice(saving_challenges)


# -------------------------
# Achievement Badges
# -------------------------

badges = {
    "🌱 Beginner Saver":
        "Started learning about budgeting!",

    "💰 Money Tracker":
        "Tracked your expenses consistently!",

    "🎯 Goal Chaser":
        "Reached a savings milestone!",

    "🧠 Mindful Spender":
        "Reduced impulse purchases!",

    "🏆 Budget Master":
        "Stayed within budget for a whole month!"
}

def show_badges():
    text = ""

    for badge, desc in badges.items():
        text += f"### {badge}\n{desc}\n\n"

    return text


# -------------------------
# Money Journal
# -------------------------

journal_entries = []

def save_journal(entry):

    if entry.strip() == "":
        return "Please write something first."

    today = datetime.date.today()

    journal_entries.append(
        f"{today} : {entry}"
    )

    return "✅ Journal entry saved!"


# -------------------------
# Financial Resources
# -------------------------

resources = """
## 📚 Financial Resources

### Budget Templates
• Monthly Budget Template
• Weekly Spending Planner
• Student Budget Sheet

### Saving Guides
• Build an Emergency Fund
• Needs vs Wants
• Saving for College
• 50/30/20 Budget Rule

### Useful Websites
• https://www.investopedia.com
• https://www.khanacademy.org
• https://www.consumer.gov
• https://www.practicalmoneyskills.com

### Helpful Apps
• Goodbudget
• Splitwise
• YNAB
• PocketGuard
"""


# -------------------------
# Emergency Help
# -------------------------

emergency_help = """
## 🚨 Emergency Financial Help

### Student Financial Aid

• Scholarships
• School Financial Office
• Government Student Aid

### Debt Counseling

If debt becomes overwhelming, contact a certified nonprofit financial counselor.

### Financial Advisors

Speak with your school's financial aid office or a trusted financial advisor before making major financial decisions.
"""

# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
