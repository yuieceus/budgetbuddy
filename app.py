import random
import datetime
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from theme import sage_theme, ocean_theme, cherry_theme

# ==========================================
# 1. RAG & KNOWLEDGE BASE SETUP
# ==========================================

# Read knowledge base file
with open("knowledge.txt", "r", encoding="utf-8") as file:
    knowledge_base = file.read()


def preprocess_text(text):
    cleaned_text = text.strip()
    chunks = cleaned_text.split("\n")
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned_chunks


cleaned_chunks = preprocess_text(knowledge_base)

# Load SentenceTransformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def create_embeddings(text_chunks):
    chunk_embeddings = embed_model.encode(text_chunks, convert_to_tensor=True)
    return chunk_embeddings


chunk_embeddings = create_embeddings(cleaned_chunks)


def get_top_chunks(query, chunk_embeddings, text_chunks):
    query_embedding = embed_model.encode(query, convert_to_tensor=True)

    # Normalize vectors for cosine similarity
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
    top_indices = torch.topk(similarities, k=min(3, len(text_chunks))).indices

    top_chunks = [text_chunks[i] for i in top_indices]
    return top_chunks


# ==========================================
# 2. AI MODEL CHATBOT SETUP
# ==========================================

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
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    return response.choices[0].message.content.strip()


# ==========================================
# 3. EXTRA INTERACTIVE FEATURES & DATA
# ==========================================

budget_tips = [
    "💡 **Pay Yourself First:** Set aside your savings target as soon as you receive allowance or earnings.",
    "💡 **The 24-Hour Rule:** Wait a full day before buying non-essential items. Most impulse urges fade!",
    "💡 **Unsubscribe from Sales Emails:** Fewer sale alerts mean fewer temptations.",
    "💡 **Track Every Penny:** Small $3 daily snacks add up to nearly $100 a month!",
    "💡 **Needs vs. Wants:** Ask yourself: *'Will bad things happen if I don't buy this today?'*",
    "💡 **Name Your Savings Goals:** Saving for 'Concert Ticket' is much easier to stick to than 'Saving Money.'"
]

saving_challenges = [
    "💰 Save $5 today in your emergency jar.",
    "🚫 Buy nothing non-essential today.",
    "📒 Log every single purchase you make today.",
    "🥤 Skip buying a drink/snack today and save the cash.",
    "🍱 Pack food or snacks from home instead of buying out.",
    "🛍️ Avoid browsing shopping apps for 24 hours.",
    "🎯 Transfer $10 straight into your savings account."
]

badges = {
    "🌱 Beginner Saver": "Started learning about budgeting and money habits.",
    "💰 Money Tracker": "Tracked your daily spending consistently.",
    "🎯 Goal Chaser": "Created and reached a savings target.",
    "🧠 Mindful Spender": "Used the cooling-off rule to stop an impulse buy.",
    "🔥 Streak Master": "Logged multiple consecutive days of mindful spending.",
    "🏆 Budget Master": "Stayed within your planned budget for a full month."
}

journal_entries = []


def random_tip():
    return random.choice(budget_tips)


def random_challenge():
    return random.choice(saving_challenges)


def show_badges():
    text = ""
    for badge, description in badges.items():
        text += f"#### {badge}\n*{description}*\n\n"
    return text


def save_journal(entry):
    if not entry.strip():
        return "⚠️ Please write an entry before saving."
    today = datetime.date.today()
    journal_entries.append(f"{today}: {entry}")
    return "✅ Journal entry saved successfully!"


def calculate_saving_challenge(target_amount, weeks):
    if not target_amount or target_amount <= 0 or not weeks or weeks <= 0:
        return "⚠️ Please enter valid positive numbers for your goal amount and timeline."

    weekly = target_amount / weeks
    daily = weekly / 7

    return f"""
### 🎯 Your Saving Roadmap
* **Total Goal:** `${target_amount:,.2f}`
* **Timeline:** `{int(weeks)} weeks`

---
* 🗓️ **Weekly Target:** **${weekly:,.2f}** / week
* ☀️ **Daily Target:** ~**${daily:,.2f}** / day

> 💡 *Put away this small daily amount at the end of each day to reach your goal effortlessly!*
"""


def evaluate_impulse_purchase(item_name, price, feeling):
    if not item_name or not price or price <= 0:
        return "⚠️ Please provide a valid item name and price."

    hours_to_wait = 24 if price < 50 else (48 if price < 150 else 72)

    return f"""
### 🛑 Cooling-Off Recommendation: {item_name} (${price:,.2f})

* **Logged Emotion:** *{feeling}*
* **Recommended Pause:** **{hours_to_wait} Hours**

---
Shopping while feeling **{feeling.lower()}** often leads to temporary comfort rather than lasting value.

> ⏱️ *Put this item on a wishlist and revisit it in **{hours_to_wait} hours**. If you still want it then and it fits your budget, go for it!*
"""


def update_streak(current_streak, logged_today):
    if logged_today == "Yes (Only essential needs)":
        new_streak = current_streak + 1
        return new_streak, f"🔥 **Awesome! Your streak is now {new_streak} day(s)!**"
    else:
        return 0, "🔄 **Streak reset.** Habit building takes practice. Start fresh tomorrow!"


# --- MARKDOWN CONTENT WITH VERIFIED DIRECT LINKS ---

articles_markdown = """
## 📰 Essential Financial Articles for Teens

### 📖 Recommended Reads
1. **[What is Financial Literacy? (Khan Academy)](https://www.khanacademy.org/college-careers-more/financial-literacy/xa6995ea67a8e9fdd:welcome-to-financial-literacy/xa6995ea67a8e9fdd:intro-to-financial-literacy/a/what-is-financial-literacy)**
   * *Overview:* An introduction to mastering everyday money decisions, understanding credit, and building confidence with personal finances.

2. **[The 50/30/20 Budgeting Rule Explained (Investopedia)](https://www.investopedia.com/ask/answers/022916/what-502030-budget-rule.asp)**
   * *Overview:* Learn how to divide your allowance or earnings between Needs (50%), Wants (30%), and Savings (20%) for balanced spending.

3. **[Crush Impulse Buying With 4 Mind Tricks (NerdWallet / MPOWER)](https://www.mpowerfinancing.com/blog/crush-impulse-buying-4-jedi-mind-tricks)**
   * *Overview:* Practical psychological tactics to overcome emotional spending triggers, flash sale traps, and retail marketing pressure.

4. **[An Essential Guide to Building an Emergency Fund (CFPB)](https://www.consumerfinance.gov/an-essential-guide-to-building-an-emergency-fund/)**
   * *Overview:* Official step-by-step guidance from the Consumer Financial Protection Bureau on creating a cash safety net for unexpected costs.
"""

resources_markdown = """
## 📚 Financial Resources & Useful Tools

### 🌐 Direct Websites
* **[Investopedia Financial Terms](https://www.investopedia.com)** — Beginner-friendly dictionary for credit, loans, and banking.
* **[Khan Academy Financial Literacy](https://www.khanacademy.org/college-careers-more/financial-literacy)** — Free interactive video modules and quizzes.
* **[Practical Money Skills](https://www.practicalmoneyskills.com)** — Educational budget games, calculators, and downloadable worksheets.

### 📱 Recommended Budgeting Apps
* **[Goodbudget](https://goodbudget.com)** — Virtual envelope system tailored for visual spenders.
* **[Splitwise](https://www.splitwise.com)** — Organizes and splits shared outings or expenses with friends.
* **[YNAB (You Need A Budget)](https://www.youneedabudget.com)** — Proactive category spending tool.
* **[PocketGuard](https://pocketguard.com)** — Automatically calculates disposable cash after essential bills.
"""

emergency_markdown = """
## 🚨 Emergency Financial Support & Helplines

### 🎓 Student Financial Assistance
* **[Federal Student Aid (FAFSA)](https://studentaid.gov)**
  * **Website:** [studentaid.gov](https://studentaid.gov)
  * **Hotline:** `1-800-4-FED-AID` (1-800-433-3243)
  * *Info:* Grants, scholarships, work-study programs, and student loan assistance.

### 📞 Non-Profit Financial Counseling
* **[National Foundation for Credit Counseling (NFCC)](https://www.nfcc.org)**
  * **Website:** [nfcc.org](https://www.nfcc.org)
  * **Free Helpline:** `1-800-388-2227`
  * *Info:* Free and confidential non-profit financial and debt management counseling.

### 🆘 Mental Health & Crisis Support
* **[Crisis Text Line](https://www.crisistextline.org)**
  * **Contact:** Text **HOME** to **741741** *(Free, 24/7, Confidential)*
  * *Info:* Immediate text support if financial stress or personal anxiety feels overwhelming.
"""


# ==========================================
# 4. CUSTOM CSS STYLING
# ==========================================

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

.section-title {
    margin-top: 24px;
    margin-bottom: 12px;
    color: #0f766e;
    border-left: 4px solid #0d9488;
    padding-left: 12px;
    font-size: 1.3rem;
    font-weight: 700;
}

.gr-accordion {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
    margin-bottom: 12px !important;
}

.gr-button {
    border-radius: 10px !important;
}

.gr-box, .gr-input {
    border-radius: 10px !important;
}
"""


# ==========================================
# 5. COMBINED GRADIO INTERFACE
# ==========================================

with gr.Blocks(theme=sage_theme, css=custom_css, title="BudgetBuddy") as chatbot:

    # Header Image
    gr.Image("BudgetBuddy_Image_Header1.png", show_label=False)

    # Theme Switcher Bar
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

    # AI Chatbot Interface
    gr.ChatInterface(
        respond,
        title="Hi, I'm BudgetBuddy! 💵",
        textbox=gr.Textbox(
            placeholder="Share your budget or ask me anything!"
        ),
        description="A smart chatbot that combines budgeting and mental wellness to help you spend mindfully, save better, and stress less!",
        examples=[
            "I get $500 per month, can you make me a budget?",
            "I keep buying things when I'm stressed. What should I do?",
            "I spent $40 on food and $60 on clothes. Can you analyze it?",
            "How can I save more?"
        ]
    )

    # Spotify Embed Player
    with gr.Row():
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

    # Extra Features Section Header
    gr.HTML("<h2 class='section-title'>✨ Interactive Toolkit & Resources</h2>")

    # Bottom Extra Features Hub
    with gr.Tabs():

        # TAB 1: SMART TOOLS
        with gr.Tab("🛠️ Smart Tools"):
            
            with gr.Accordion("🎯 Savings Goal Roadmap", open=True):
                gr.Markdown("Break large savings goals into achievable weekly and daily targets.")
                with gr.Row():
                    with gr.Column():
                        goal_name = gr.Textbox(label="Goal Name", placeholder="e.g., Concert Ticket, New Headphones")
                        goal_amount = gr.Number(label="Target Amount ($)", value=100.0, precision=2)
                        timeline_weeks = gr.Slider(minimum=1, maximum=52, value=4, step=1, label="Timeline (Weeks)")
                        calc_btn = gr.Button("Calculate Roadmap", variant="primary")
                    with gr.Column():
                        roadmap_output = gr.Markdown("Enter your goal details to calculate your roadmap.")

                calc_btn.click(
                    fn=calculate_saving_challenge,
                    inputs=[goal_amount, timeline_weeks],
                    outputs=roadmap_output
                )

            with gr.Accordion("🛑 Impulse Purchase Cooling-Off Sandbox", open=False):
                gr.Markdown("Test whether a purchase is driven by temporary emotion before buying.")
                with gr.Row():
                    with gr.Column():
                        item_input = gr.Textbox(label="Item Name", placeholder="e.g., Limited Edition Sneakers")
                        price_input = gr.Number(label="Price ($)", value=45.0, precision=2)
                        emotion_input = gr.Dropdown(
                            choices=["Bored", "Stressed", "Excited", "Sad", "FOMO (Peer Pressure)", "Neutral"],
                            value="Bored",
                            label="Current Emotion"
                        )
                        eval_btn = gr.Button("Evaluate Purchase", variant="primary")
                    with gr.Column():
                        cooling_output = gr.Markdown("Enter purchase details to receive a recommendation.")

                eval_btn.click(
                    fn=evaluate_impulse_purchase,
                    inputs=[item_input, price_input, emotion_input],
                    outputs=cooling_output
                )

            with gr.Accordion("🔥 No-Spend Day Streak Tracker", open=False):
                gr.Markdown("Track your consecutive days of spending money **only** on essential needs.")
                streak_state = gr.State(value=0)
                with gr.Row():
                    spend_radio = gr.Radio(
                        choices=["Yes (Only essential needs)", "No (Bought non-essentials/impulse items)"],
                        label="Did you stick to essential spending today?"
                    )
                    streak_btn = gr.Button("Log Today")

                streak_msg = gr.Markdown("Current Streak: **0 Days**")
                streak_btn.click(
                    fn=update_streak,
                    inputs=[streak_state, spend_radio],
                    outputs=[streak_state, streak_msg]
                )

        # TAB 2: DAILY SPARKS & JOURNAL
        with gr.Tab("🌟 Daily Sparks & Journal"):

            with gr.Accordion("💡 Budget Tip of the Day", open=True):
                tip_out = gr.Markdown(random_tip())
                gr.Button("🎲 Get Random Tip", variant="secondary").click(fn=random_tip, outputs=tip_out)

            with gr.Accordion("🎯 Daily Saving Challenge", open=False):
                challenge_out = gr.Markdown(random_challenge())
                gr.Button("🎲 Get New Challenge", variant="secondary").click(fn=random_challenge, outputs=challenge_out)

            with gr.Accordion("🏆 Achievement Badges", open=False):
                badge_box = gr.Markdown(show_badges())

            with gr.Accordion("📖 Mindful Money Journal", open=False):
                journal_input = gr.Textbox(lines=4, placeholder="Write about today's spending, feelings, or financial goals...")
                status_out = gr.Markdown()
                gr.Button("Save Journal Entry", variant="primary").click(
                    fn=save_journal,
                    inputs=journal_input,
                    outputs=status_out
                )

        # TAB 3: GUIDES, ARTICLES & HELP
        with gr.Tab("📚 Guides, Articles & Help"):

            with gr.Accordion("📰 Essential Financial Articles", open=True):
                gr.Markdown(articles_markdown)

            with gr.Accordion("📚 Financial Resources & Useful Tools", open=False):
                gr.Markdown(resources_markdown)

            with gr.Accordion("🚨 Emergency Financial Support & Helplines", open=False):
                gr.Markdown(emergency_markdown)

# Launch app
if __name__ == "__main__":
    chatbot.launch()