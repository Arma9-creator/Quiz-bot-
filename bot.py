import os
import re
import json
import random
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Dict

from telegram import (
    Update,
    Poll,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ================= CONFIG =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))  # OCR first N pages
OCR_ZOOM = float(os.getenv("OCR_ZOOM", "2.5"))

# ================= DATA MODELS =================
@dataclass
class MCQ:
    qid: str
    question: str
    options: List[str]
    correct_index: int

@dataclass
class Store:
    mcqs: List[MCQ]
    sessions: Dict[int, dict]

def get_store(context: ContextTypes.DEFAULT_TYPE) -> Store:
    if "store" not in context.bot_data:
        context.bot_data["store"] = Store(mcqs=[], sessions={})
    return context.bot_data["store"]

# ================= OCR =================
def page_to_image(doc, i):
    page = doc.load_page(i)
    mat = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_two_columns(img: Image.Image) -> str:
    w, h = img.size
    left = img.crop((0, 0, w//2, h))
    right = img.crop((w//2, 0, w, h))
    text = pytesseract.image_to_string(left) + "\n" + pytesseract.image_to_string(right)
    text = re.sub(r"JOIN\s*-\s*@\S+", "", text, flags=re.I)
    return text

# ================= PARSING =================
MCQ_RE = re.compile(
    r"(.*?)\(\s*a\s*\)(.*?)\(\s*b\s*\)(.*?)\(\s*c\s*\)(.*?)\(\s*d\s*\)(.*)",
    re.S | re.I
)

def extract_mcqs(text: str) -> List[MCQ]:
    mcqs = []
    qno = 1
    blocks = re.split(r"\n\s*\d+\.\s*", text)
    for block in blocks:
        m = MCQ_RE.search(block)
        if not m:
            continue
        q = re.sub(r"\s+", " ", m.group(1)).strip()
        opts = [m.group(i).strip() for i in range(2, 6)]
        if len(q) > 10:
            mcqs.append(MCQ(
                qid=str(qno),
                question=q,
                options=opts,
                correct_index=0  # default, can improve later
            ))
            qno += 1
    return mcqs

# ================= BOT COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 PDF Quiz Bot Ready\n\n"
        "1️⃣ Send your scanned PDF\n"
        "2️⃣ Use /mcq to start quiz\n\n"
        "MCQ supports:\n"
        "• Practice (serial)\n"
        "• Test (random + marks)"
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    store = get_store(context)
    await update.message.reply_text("📄 PDF received. Running OCR…")

    file = await update.message.document.get_file()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "input.pdf")
        await file.download_to_drive(path)

        doc = fitz.open(path)
        texts = []
        for i in range(min(len(doc), MAX_PAGES)):
            img = page_to_image(doc, i)
            texts.append(ocr_two_columns(img))

        text = "\n".join(texts)
        store.mcqs = extract_mcqs(text)

    await update.message.reply_text(
        f"✅ Extracted {len(store.mcqs)} MCQs\n\n"
        "Use /mcq to start"
    )

# ================= MCQ MENU =================
def count_keyboard(prefix, max_n, page=0):
    per_page = 16
    start = 5 + page * per_page
    end = min(max_n, start + per_page - 1)

    rows, row = [], []
    for i in range(start, end + 1):
        row.append(InlineKeyboardButton(str(i), callback_data=f"{prefix}{i}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    nav = []
    if start > 5:
        nav.append(InlineKeyboardButton("⬅ Prev", callback_data=f"{prefix}page:{page-1}"))
    if end < max_n:
        nav.append(InlineKeyboardButton("Next ➡", callback_data=f"{prefix}page:{page+1}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("Cancel", callback_data=f"{prefix}cancel")])
    return InlineKeyboardMarkup(rows)

async def mcq_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    store = get_store(context)
    if not store.mcqs:
        await update.message.reply_text("Upload a PDF first.")
        return

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📘 Practice (Serial)", callback_data="mcqmode:practice")],
        [InlineKeyboardButton("📝 Test (Random + Marks)", callback_data="mcqmode:test")]
    ])
    await update.message.reply_text(
        f"Total MCQs: {len(store.mcqs)}\nChoose mode:",
        reply_markup=kb
    )

async def mcq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    store = get_store(context)
    data = query.data

    if data.startswith("mcqmode:"):
        mode = data.split(":")[1]
        kb = count_keyboard(f"mcq|{mode}|", len(store.mcqs))
        await query.edit_message_text(
            f"Mode: {mode.upper()}\nChoose number of questions:",
            reply_markup=kb
        )
        return

    if data.startswith("mcq|"):
        _, mode, tail = data.split("|")
        if tail == "cancel":
            await query.edit_message_text("Cancelled.")
            return

        if tail.startswith("page:"):
            page = int(tail.split(":")[1])
            kb = count_keyboard(f"mcq|{mode}|", len(store.mcqs), page)
            await query.edit_message_text(
                f"Mode: {mode.upper()}\nChoose number of questions:",
                reply_markup=kb
            )
            return

        n = int(tail)
        await query.edit_message_text(f"Starting {mode.upper()} MCQ quiz ({n} questions)…")
        await start_mcq(query.message.chat_id, mode, n, context)

async def start_mcq(chat_id, mode, n, context):
    store = get_store(context)
    qs = store.mcqs[:]
    if mode == "test":
        random.shuffle(qs)
    qs = qs[:n]

    for q in qs:
        await context.bot.send_poll(
            chat_id=chat_id,
            question=q.question[:300],
            options=[o[:100] for o in q.options],
            type=Poll.QUIZ,
            correct_option_id=q.correct_index,
            is_anonymous=False
        )

# ================= MAIN =================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mcq", mcq_menu))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(CallbackQueryHandler(mcq_callback))

    app.run_polling()

if __name__ == "__main__":
    main()
