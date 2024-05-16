from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext
from telegram.ext import filters
import os
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_pdf(uploaded_file):
    # Create the temporary directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Save the uploaded file to the temporary location
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the PDF file
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    # Remove the temporary file
    os.remove(file_path)

    return text

def summarize_long_pdf(text):
    llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-pro")
    summary = llm(text)
    return summary

def echo(update: Update, context: CallbackContext) -> None:
    # Get the user's message
    user_message = update.message.text

    if user_message.startswith("/pdf_summary"):
        # Process PDF summary request
        pdf_file = context.user_data.get("pdf_file")
        if pdf_file:
            pdf_text = read_pdf(pdf_file)
            summary = summarize_long_pdf(pdf_text)
            update.message.reply_text(summary)
        else:
            update.message.reply_text("Please upload a PDF file first.")
    else:
        # Echo the user's message back
        update.message.reply_text("You said: " + user_message)

def main() -> None:
    # Create the Updater and pass it your bot's token
    updater = Updater(tokens="6911636949:AAHhxaA95X88Oa8KfMaBV5OUD9WQKABAtOs",use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register command handler for /pdf_summary command
    dispatcher.add_handler(CommandHandler("pdf_summary", echo))

    # Register a message handler for regular text messages
    dispatcher.add_handler(MessageHandler(filters.text & ~filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C
    updater.idle()

if __name__ == '__main__':
    main()
