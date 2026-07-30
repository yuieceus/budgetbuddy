import random
import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
from theme import sage_theme, ocean_theme, cherry_theme
#The three extra features
saving_challenges = [
    "💰 Save $5 today in your emergency jar.",
    "🚫 Buy nothing non-essential today.",
    "📒 Log every single purchase you make today.",
    "🥤 Skip buying a drink/snack today and save the cash.",
    "🍱 Pack food or snacks from home instead of buying out.",
    "🛍️ Avoid browsing shopping apps for 24 hours.",
    "🎯 Transfer $10 straight into your savings account."
]

def random_challenge():
    return random.choice(saving_challenges)

def update_streak(current_streak, logged_today):
    if logged_today == "Yes (Only essential needs)":
        new_streak = current_streak + 1
        return new_streak, f"🔥 **Awesome! Your streak is now {new_streak} day(s)!**"
    else:
        return 0, "🔄 **Streak reset.** Habit building takes practice. Start fresh tomorrow!"

resources_markdown = """
## 📚 Financial Resources & Useful Tools

### 🌐 Educational Websites
* **[Investopedia](https://www.investopedia.com):** Direct definitions and beginner financial guides.
* **[Khan Academy Financial Literacy](https://www.khanacademy.org/college-careers-more/financial-literacy):** Free interactive courses on budgeting, banking, and consumer credit.
* **[Practical Money Skills](https://www.practicalmoneyskills.com):** Financial games, budget calculators, and downloadable student guides.

### 📱 Helpful Budgeting Apps
* **[Goodbudget](https://goodbudget.com):** Envelope-style budgeting tool ideal for visual spenders.
* **[Splitwise](https://www.splitwise.com):** Easily manage and split group expenses with friends.
"""

emergency_markdown = """
## 🚨 Emergency Financial Help & Direct Support

### 🎓 Student Aid & Academic Hardship
* **[Federal Student Aid (FAFSA)](https://studentaid.gov)**
  * *Website:* [studentaid.gov](https://studentaid.gov)
  * *Contact:* **1-800-4-FED-AID** (1-800-433-3243)
  * *Info:* Guidance on federal grants, scholarships, work-study programs, and student aid eligibility.

### 📞 Debt Counseling & Financial Stress
* **[National Foundation for Credit Counseling (NFCC)](https://www.nfcc.org)**
  * *Website:* [nfcc.org](https://www.nfcc.org)
  * *Toll-Free Helpline:* **1-800-388-2227**
  * *Info:* Free and low-cost non-profit counseling for managing debt and financial stress.

### 🆘 Mental Health & Crisis Support
* **[Crisis Text Line](https://www.crisistextline.org)**
  * *Contact:* Text **HOME** to **741741** (Free, 24/7, Confidential)
  * *Info:* Instant text support if money anxiety or emotional stress becomes overwhelming.
"""

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
    You are Budget Buddy, a supportive financial wellness chatbot for teenagers.
    Your personality:
    - Sound like a friendly older sibling, not a teacher.
    - Be warm, encouraging, and conversational.
    - Use simple language.
    - Never lecture or judge the user.
    - Focus on progress, not perfection.
    - Validate the user's feelings before giving advice.
    - Keep responses under 100 words.
    - End with one helpful follow-up question when appropriate.
    - Do NOT copy the knowledge word-for-word. Explain it naturally.
    Use ONLY the information below to answer the user's question.
    If the answer isn't in the knowledge, say:
    "I don't know based on the provided knowledge."
    
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
        temperature=0.7,
        top_p=0.9
    )
    return response['choices'][0]['message']['content'].strip() 

custom_css = """
.right-top-bar {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: 12px;
}

.right-top-bar button {
    background-color: var(--btn-bg, #2F4F3E) !important;
    color: white !important;
    border-radius: 20px !important;
    border: none !important;
    padding: 6px 14px !important;
    font-size: 13px !important;
}

.message.user {
    background-color: #F3F4F6 !important;
    color: #1F2937 !important;
    border: 1px solid #E5E7EB !important;
}

.message.bot {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #E5E7EB !important;
}
"""

with gr.Blocks(theme=sage_theme, css=custom_css) as chatbot:
    gr.Image("BudgetBuddy_Image_Header1.png", show_label=False)
    with gr.Row(elem_classes=["right-top-bar"]):
        btn_sage = gr.Button("🌿 Sage Garden")
        btn_ocean = gr.Button("🌊 Ocean Breeze")
        btn_cherry = gr.Button("🌸 Cherry Blossom")
        
        btn_sage.click(
        fn=None, 
        js="""() => { 
            document.body.style.backgroundColor = '#F7FBF5'; 
            document.documentElement.style.setProperty('--btn-bg', '#2F4F3E');
        }"""
    )

        btn_ocean.click(
        fn=None, 
        js="""() => { 
            document.body.style.backgroundColor = '#F0F8FF'; 
            document.documentElement.style.setProperty('--btn-bg', '#1E3A8A');
        }"""
    )

        btn_cherry.click(
        fn=None, 
        js="""() => { 
            document.body.style.backgroundColor = '#FFF5F7'; 
            document.documentElement.style.setProperty('--btn-bg', '#831843');
        }"""
    )

    gr.ChatInterface(
        respond,
        title="Hi, I'm BudgetBuddy! 💵",
        textbox=gr.Textbox(
            placeholder="Share your budget or ask me anything!"
        ),
        description="""A smart chatbot that combines budgeting and mental wellness to help you spend mindfully, save better, and stress less!

        Disclaimer: This chatbot is not a licensed therapist or financial advisor and is not intended to provide professional mental health care or personalized financial advice.""",

        examples=[
            "I get $500 per month, can you make me a budget?",
            "I keep buying things when I'm stressed. What should I do?",
            "I spent $40 on food and $60 on clothes. Can you analyze it?",
            "How can I save more?"
        ]
    )
    gr.Markdown("---")
    gr.Markdown("### 🎯 Daily Saving Challenge")
    challenge_box = gr.Textbox(label="Your Challenge", value=random_challenge, interactive=False)
    btn_new_challenge = gr.Button("🎲 Get Another Challenge")
    btn_new_challenge.click(fn=random_challenge, outputs=challenge_box)
    
    gr.Markdown("---")
    gr.Markdown("### 🔥 Daily Habit Tracker")
    streak_count = gr.State(value=0)
    logged_radio = gr.Radio(
        choices=["Yes (Only essential needs)", "No (Made non-essential purchases)"],
        label="Did you stick to essential spending today?"
    )
    btn_streak = gr.Button("Log Today's Progress")
    streak_output = gr.Markdown()
    btn_streak.click(fn=update_streak, inputs=[streak_count, logged_radio], outputs=[streak_count, streak_output])
    
    gr.Markdown("---")
    gr.Markdown(resources_markdown)
    
    gr.Markdown("---")
    gr.Markdown(emergency_markdown)
    
    gr.Markdown("---")
    gr.HTML(
            """
            <iframe style="border-radius:12px"
                    src="https://open.spotify.com/embed/track/6xsOIolcDvXCHyJkpWJVuk?utm_source=generator&theme=0&si=4c9098d8178441da"
                    width="100%"
                    height="152"
                    frameBorder="0"
                    allowfullscreen=""
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                    loading="lazy">
            </iframe>
            """
        )
        
chatbot.launch()
# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own! 