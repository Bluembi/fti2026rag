import os
import chromadb
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


DB_PATH = "./fact_db"
MODEL_ID = "tiiuae/Falcon-H1-1.5B-Instruct"
TELEGRAM_TOKEN = os.getenv("TG_TOKEN")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="user_facts")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device="cpu"
)


async def add_fact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fact = " ".join(context.args)
    if not fact:
        await update.message.reply_text("Ввод факта: /add [fact]")
        return
    
    collection.add(
        documents=[fact],
        ids=[str(update.message.message_id)]
    )
    await update.message.reply_text(f"Сохранено")

async def list_facts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    results = collection.get()
    facts = results.get('documents', [])
    if not facts:
        await update.message.reply_text("БД пуста.")
    else:
        response = "\n".join([f"• {f}" for f in facts])
        await update.message.reply_text(f"Факты:\n{response}")

async def rag_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.replace('/rag ', '')
    results = collection.query(query_texts=[query], n_results=2)
    retrieved_context = " ".join(results['documents'][0])
    
    prompt = f"Контекст: {retrieved_context}\n\nВопрос: {query}\n\nОтвет:"
    
    output = pipe(prompt, max_new_tokens=150)
    await update.message.reply_text(output[0]['generated_text'])

async def direct_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.replace('/chat ', '')
    output = pipe(query, max_new_tokens=150)
    await update.message.reply_text(output[0]['generated_text'])

async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = '''
    /add - Команда для добавления факта в ChromaDB
    /list - Показывает список всех фактов
    /rag - Запрос к модели с учетом сохраненных фактов
    /chat - Запрос к модели без учета фактов
    '''
    await update.message.reply_text(message)


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("add", add_fact))
    app.add_handler(CommandHandler("list", list_facts))
    app.add_handler(CommandHandler("rag", rag_query))
    app.add_handler(CommandHandler("chat", direct_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, show_help))
    
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()