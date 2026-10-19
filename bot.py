"""
bot.py - Telegram Medical Mind Map Markdown Generator Bot (v2.0 High-Speed Edition)
===================================================================================
طراحی‌شده با معماری هم‌ساختار با `mindmapbotargon-1.0.0`:
- دریافت متن‌های بسیار ساده، جملات کوتاه، جزوات خام یا فایل‌های متنی (.txt / .md)
- اتصال موازی و فوق‌سریع به هوش مصنوعی با کلید API شما (`AQ.Ab8RN6...`) بدون خطای Timeout
- تولید فایل استاندارد `.md` و `.opml` صددرصد سازگار با ربات رندرینگ نقشه ذهنی
- قابلیت ویرایش تعاملی نقشه ذهنی با دستور متنی و انتخاب از میان ۶ الگوی استاندارد مدوفست
"""

from __future__ import annotations

import asyncio
import html
import io
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env_file = Path(".env")
    if _env_file.is_file():
        for _line in _env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from ai_engine import (
    DEFAULT_AI_API_KEY,
    ENRICH_MODES,
    TEMPLATE_MODES,
    generate_medical_mindmap_markdown,
    get_active_api_key,
    save_api_key_to_env,
)
from keep_alive import keep_alive
from parser import Node, tree_to_opml

BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
    or ""
).strip()
AI_API_KEY = get_active_api_key()
MAX_CONCURRENT_AI_CALLS = int(os.getenv("MAX_CONCURRENT_AI_CALLS", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("mindmap_md_bot")

WAIT_INPUT = 0
WAIT_MODE = 1
WAIT_EDIT_TEXT = 2

_CB_GEN = "GEN:"
_CB_ACT = "ACT:"
_SESSIONS_KEY = "_md_sessions"
_MAX_SESSIONS_PER_USER = 8
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
_AI_TIMEOUT_SEC = 180.0

_AI_SEMAPHORE = asyncio.Semaphore(max(1, MAX_CONCURRENT_AI_CALLS))


def _esc(value: object) -> str:
    return html.escape(str(value))


def _safe_filename(title: str, suffix: str = ".md") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)[:60].strip("_")
    if not cleaned:
        cleaned = "medical_mindmap"
    return f"{cleaned}{suffix}"


def _store_session(
    user_data: Dict[str, Any],
    raw_input: str,
    source_name: str,
    markdown_text: str = "",
    tree: Optional[Node] = None,
    template_mode: str = "auto",
    enrich_mode: str = "enrich",
) -> str:
    store = user_data.get(_SESSIONS_KEY)
    if not isinstance(store, dict):
        store = {}
        user_data[_SESSIONS_KEY] = store

    while len(store) >= _MAX_SESSIONS_PER_USER:
        oldest_key = next(iter(store))
        store.pop(oldest_key, None)

    token = uuid.uuid4().hex[:12]
    store[token] = {
        "raw_input": raw_input,
        "source_name": source_name,
        "markdown_text": markdown_text,
        "tree": tree,
        "template_mode": template_mode,
        "enrich_mode": enrich_mode,
    }
    return token


def _get_session(user_data: Dict[str, Any], token: str) -> Optional[Dict[str, Any]]:
    store = user_data.get(_SESSIONS_KEY)
    if not isinstance(store, dict):
        return None
    entry = store.get(token)
    return entry if isinstance(entry, dict) else None


async def _safe_send(coro_factory, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except RetryAfter as exc:
            if attempt >= max_retries:
                raise
            wait_sec = float(exc.retry_after) + 1.0
            await asyncio.sleep(wait_sec)


def build_mode_selection_keyboard(token: str) -> InlineKeyboardMarkup:
    """کیبورد انتخاب الگوی ساختاری نقشه ذهنی پزشکی و سطح تکمیل علمی."""
    rows = [
        [
            InlineKeyboardButton(
                text="🚀 ساخت فوری (تشخیص خودکار الگو + تکمیل علمی)",
                callback_data=f"{_CB_GEN}{token}:auto:enrich",
            )
        ],
        [
            InlineKeyboardButton(
                text="🔒 ساخت وفادار به متن (بدون افزودن مطالب خارج از متن)",
                callback_data=f"{_CB_GEN}{token}:auto:strict",
            )
        ],
        [
            InlineKeyboardButton(
                text="🦠 الگوی بیماری تک‌موضوعی",
                callback_data=f"{_CB_GEN}{token}:disease:enrich",
            ),
            InlineKeyboardButton(
                text="🧭 الگوی اپروچ / فلوچارت",
                callback_data=f"{_CB_GEN}{token}:approach:enrich",
            ),
        ],
        [
            InlineKeyboardButton(
                text="🗂 دسته‌بندی اتیولوژی / DDx",
                callback_data=f"{_CB_GEN}{token}:etiology:enrich",
            ),
            InlineKeyboardButton(
                text="🔎 وورک‌آپ و علائم هشدار",
                callback_data=f"{_CB_GEN}{token}:workup:enrich",
            ),
        ],
        [
            InlineKeyboardButton(
                text="📚 مقایسه چند بیماری/اختلال",
                callback_data=f"{_CB_GEN}{token}:multidisorder:enrich",
            ),
            InlineKeyboardButton(
                text="🧠 معاینه و آناتومی بالینی",
                callback_data=f"{_CB_GEN}{token}:exam:enrich",
            ),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def build_post_generation_keyboard(token: str) -> InlineKeyboardMarkup:
    """کیبورد عملیات پس از تولید فایل Markdown."""
    rows = [
        [
            InlineKeyboardButton(
                text="✏️ ویرایش هوشمند با دستور متنی",
                callback_data=f"{_CB_ACT}{token}:edit",
            ),
            InlineKeyboardButton(
                text="📦 دریافت فایل OPML",
                callback_data=f"{_CB_ACT}{token}:opml",
            ),
        ],
        [
            InlineKeyboardButton(
                text="🔄 تغییر الگو / بازسازی مجدد",
                callback_data=f"{_CB_ACT}{token}:reselect",
            ),
        ],
    ]
    return InlineKeyboardMarkup(rows)


async def _prompt_mode_selection(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    raw_input: str,
    source_label: str,
) -> int:
    token = _store_session(
        user_data=context.user_data,
        raw_input=raw_input,
        source_name=source_label,
    )
    preview_snippet = raw_input.strip().replace("\n", " ")
    if len(preview_snippet) > 120:
        preview_snippet = preview_snippet[:120] + "…"

    msg_text = (
        f"✅ ورودی <b>{_esc(source_label)}</b> دریافت شد!\n\n"
        f"📝 <b>پیش‌نمایش ورودی:</b> <code>{_esc(preview_snippet)}</code>\n\n"
        "لطفاً <b>نحوه ساختاردهی و الگوی نقشه ذهنی پزشکی</b> را انتخاب کنید:"
    )
    await update.effective_message.reply_text(
        text=msg_text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_mode_selection_keyboard(token),
    )
    return WAIT_MODE


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    welcome_lines = [
        "👋 سلام! به <b>ربات هوشمند تولیدکننده Markdown نقشه ذهنی پزشکی (سبک مدوفست — نسخه سریع v2.0)</b> خوش آمدید.",
        "",
        "من متن‌های بسیار ساده، جملات کوتاه، جزوات خام یا حتی <b>فقط نام یک بیماری یا مبحث پزشکی</b> را می‌گیرم و آن را به <b>فایل استاندارد <code>.md</code> و <code>.opml</code></b> (صددرصد آماده برای ارسال به ربات رندرینگ نقشه ذهنی شما) تبدیل می‌کنم.",
        "",
        "🧠 <b>ویژگی‌های تخصصی این ربات:</b>",
        "• تسلط کامل بر ۶ الگوی استاندارد نقشه‌های پزشکی (بیماری تک‌موضوعی، اپروچ بالینی، دسته‌بندی اتیولوژی، وورک‌آپ تشخیصی، چند بیماری، و معاینه بالینی)",
        "• رعایت دقیق سلسله‌مراتب <code>#</code> (موضوع مرکزی)، <code>##</code> (شاخه‌های اصلی) و <code>-</code> (زیرشاخه‌ها با تورفتگی استاندارد ۲ فاصله)",
        "• حفظ هوشمندانه اعداد، درصدها، بازه‌های زمانی، کات‌آف‌های آزمایشگاهی و مراحل درمانی (قدم اول، قدم دوم...)",
        "",
        "📥 <b>کافیست همین الان یکی از موارد زیر را بفرستید:</b>",
        "1️⃣ یک جمله، پاراگراف ساده یا حتی نام یک مبحث (مثلاً: <code>DIC</code> یا <code>اپروچ به زردی نوزاد</code>)",
        "2️⃣ فایل متنی <code>.txt</code> یا <code>.md</code>",
        "",
        "برای لغو عملیات در هر زمان: /cancel",
    ]
    await update.effective_message.reply_text(
        text="\n".join(welcome_lines),
        parse_mode=ParseMode.HTML,
    )
    return WAIT_INPUT


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.effective_message.reply_text(
        text="❌ عملیات لغو شد. هر زمان مایل بودید متن، موضوع یا فایل جدیدی ارسال کنید.",
        parse_mode=ParseMode.HTML,
    )
    return ConversationHandler.END


async def cmd_setkey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """تنظیم و ذخیره فوری کلید API هوش مصنوعی با دستور /setkey"""
    global AI_API_KEY
    message = update.effective_message
    if not message:
        return WAIT_INPUT

    args_text = " ".join(context.args).strip() if context.args else ""
    if not args_text and message.text:
        parts = message.text.strip().split(maxsplit=1)
        if len(parts) > 1:
            args_text = parts[1].strip()

    if not args_text:
        await message.reply_text(
            text=(
                "🔑 <b>نحوه تنظیم کلید هوش مصنوعی:</b>\n\n"
                "لطفاً کلید خود را بعد از دستور <code>/setkey</code> ارسال کنید؛ مثال:\n"
                "<code>/setkey AQ.Ab8RN6KWuLT-fUlCygajyDzunN7T21lP-RBU7Y3t_uOOv5dgAQ</code>\n\n"
                "یا کلید Google AI Studio (شروع با <code>AQ.</code> یا <code>AIza</code>) یا CodeCraft (شروع با <code>cc_</code>) را مستقیماً در چت بفرستید."
            ),
            parse_mode=ParseMode.HTML,
        )
        return WAIT_INPUT

    if save_api_key_to_env(args_text):
        AI_API_KEY = get_active_api_key(args_text)
        masked = f"{AI_API_KEY[:6]}...{AI_API_KEY[-4:]}" if len(AI_API_KEY) > 12 else AI_API_KEY
        provider = "CodeCraft API" if AI_API_KEY.startswith("cc_") else ("Google AI Studio (Gemini)" if AI_API_KEY.startswith(("AQ.", "AIza")) else "OpenAI API")
        await message.reply_text(
            text=(
                f"✅ <b>کلید هوش مصنوعی با موفقیت ذخیره و فعال شد!</b>\n"
                f"🔹 <b>سرویس:</b> <code>{_esc(provider)}</code>\n"
                f"🔑 <b>کلید فعال:</b> <code>{_esc(masked)}</code>\n\n"
                "اکنون می‌توانید مبحث پزشکی یا متن خود را برای تولید نقشه ذهنی ارسال کنید."
            ),
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.reply_text(
            text="⚠️ کلید واردشده معتبر نیست. لطفاً یک کلید معتبر ارسال کنید.",
            parse_mode=ParseMode.HTML,
        )
    return WAIT_INPUT


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.effective_message
    doc = message.document if message else None
    if not doc:
        return WAIT_INPUT

    filename = doc.file_name or "input.txt"
    ext = Path(filename).suffix.lower()

    if ext not in (".txt", ".md", ".markdown"):
        await message.reply_text(
            text="⚠️ لطفاً فایل متنی با فرمت <code>.txt</code> یا <code>.md</code> ارسال کنید (یا متن خود را مستقیماً در چت بنویسید).",
            parse_mode=ParseMode.HTML,
        )
        return WAIT_INPUT

    if doc.file_size and doc.file_size > _MAX_UPLOAD_BYTES:
        await message.reply_text(
            text="⚠️ حجم فایل ارسالی بیش از حد مجاز (۵ مگابایت) است.",
            parse_mode=ParseMode.HTML,
        )
        return WAIT_INPUT

    try:
        tg_file = await doc.get_file()
        raw_bytes = bytes(await tg_file.download_as_bytearray())
        raw_text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
    except Exception as exc:
        logger.exception("Failed to read uploaded file '%s'", filename)
        await message.reply_text(
            text=f"❌ خطا در خواندن فایل ارسالی:\n<code>{_esc(exc)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return WAIT_INPUT

    if not raw_text:
        await message.reply_text(
            text="⚠️ فایل ارسالی خالی است.",
            parse_mode=ParseMode.HTML,
        )
        return WAIT_INPUT

    return await _prompt_mode_selection(
        update=update,
        context=context,
        raw_input=raw_text,
        source_label=filename,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global AI_API_KEY
    message = update.effective_message
    raw_text = ((message.text if message else "") or "").strip()
    if not raw_text:
        return WAIT_INPUT

    # تشخیص خودکار در صورتی که کاربر کلید API را مستقیماً بدون /setkey ارسال کرده باشد (شامل AQ. و AIza و cc_ و sk-)
    if re.match(r"^(?:cc_[A-Za-z0-9_\-]{20,}|AIza[A-Za-z0-9_\-]{25,}|AQ\.[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})$", raw_text):
        if save_api_key_to_env(raw_text):
            AI_API_KEY = get_active_api_key(raw_text)
            masked = f"{AI_API_KEY[:6]}...{AI_API_KEY[-4:]}"
            provider = "Google AI Studio (Gemini)" if AI_API_KEY.startswith(("AQ.", "AIza")) else ("CodeCraft API" if AI_API_KEY.startswith("cc_") else "OpenAI API")
            await message.reply_text(
                text=(
                    f"✅ <b>کلید هوش مصنوعی شناسایی و به صورت خودکار فعال شد!</b>\n"
                    f"🔹 <b>سرویس:</b> <code>{_esc(provider)}</code>\n"
                    f"🔑 <b>کلید فعال:</b> <code>{_esc(masked)}</code>\n\n"
                    "حالا نام بیماری یا متن پزشکی خود را ارسال کنید."
                ),
                parse_mode=ParseMode.HTML,
            )
            return WAIT_INPUT

    editing_token = context.user_data.get("_active_edit_token")
    if isinstance(editing_token, str) and editing_token:
        context.user_data.pop("_active_edit_token", None)
        session = _get_session(context.user_data, editing_token)
        if session and session.get("markdown_text"):
            return await _execute_ai_generation(
                update=update,
                context=context,
                token=editing_token,
                template_mode=str(session.get("template_mode", "auto")),
                enrich_mode=str(session.get("enrich_mode", "enrich")),
                edit_instruction=raw_text,
            )

    return await _prompt_mode_selection(
        update=update,
        context=context,
        raw_input=raw_text,
        source_label="متن ارسالی",
    )


async def _execute_ai_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    token: str,
    template_mode: str,
    enrich_mode: str,
    edit_instruction: Optional[str] = None,
) -> int:
    session = _get_session(context.user_data, token)
    target_msg = update.effective_message

    if not session:
        if target_msg:
            await target_msg.reply_text(
                text="⚠️ نشست قبلی منقضی شده است. لطفاً متن یا موضوع خود را مجدداً ارسال کنید.",
                parse_mode=ParseMode.HTML,
            )
        return WAIT_INPUT

    session["template_mode"] = template_mode
    session["enrich_mode"] = enrich_mode

    mode_label = TEMPLATE_MODES.get(template_mode, TEMPLATE_MODES["auto"])["label"]
    enrich_label = ENRICH_MODES.get(enrich_mode, ENRICH_MODES["enrich"])["label"]

    status_text = (
        "⚙️ <b>در حال اعمال ویرایش هوشمند روی نقشه ذهنی…</b>"
        if edit_instruction
        else (
            f"🧠 <b>در حال آنالیز پزشکی و طراحی ساختار نقشه ذهنی…</b>\n"
            f"📐 الگو: <b>{_esc(mode_label)}</b>\n"
            f"🔬 حالت: <b>{_esc(enrich_label)}</b>"
        )
    )

    status_msg = None
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.edit_text(
                text=status_text,
                parse_mode=ParseMode.HTML,
            )
            status_msg = update.callback_query.message
        except TelegramError:
            status_msg = await target_msg.reply_text(
                text=status_text,
                parse_mode=ParseMode.HTML,
            )
    elif target_msg:
        status_msg = await target_msg.reply_text(
            text=status_text,
            parse_mode=ParseMode.HTML,
        )

    try:
        async with _AI_SEMAPHORE:
            canonical_md, tree = await asyncio.wait_for(
                generate_medical_mindmap_markdown(
                    raw_input=str(session["raw_input"]),
                    template_mode=template_mode,
                    enrich_mode=enrich_mode,
                    edit_instruction=edit_instruction,
                    previous_markdown=str(session.get("markdown_text") or "") if edit_instruction else None,
                    api_key=get_active_api_key(),
                ),
                timeout=_AI_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        logger.error("AI generation timed out after %s seconds", _AI_TIMEOUT_SEC)
        if status_msg:
            await status_msg.edit_text(
                text="⏱ زمان پاسخ‌گویی بیش از حد مجاز طول کشید. لطفاً مجدداً تلاش کنید.",
                parse_mode=ParseMode.HTML,
            )
        return WAIT_INPUT
    except Exception as exc:
        logger.exception("AI generation failed")
        if status_msg:
            await status_msg.edit_text(
                text=f"❌ خطا در تولید نقشه ذهنی:\n<code>{_esc(exc)}</code>",
                parse_mode=ParseMode.HTML,
            )
        return WAIT_INPUT

    session["markdown_text"] = canonical_md
    session["tree"] = tree

    total_nodes = tree.count_nodes()
    main_branches_count = len(tree.children)
    max_depth = tree.max_depth()
    main_titles = " ، ".join(ch.text for ch in tree.children[:7])
    if len(tree.children) > 7:
        main_titles += " و …"

    md_filename = _safe_filename(tree.text, suffix=".md")

    if target_msg:
        await _safe_send(
            lambda: target_msg.reply_document(
                document=InputFile(
                    io.BytesIO(canonical_md.encode("utf-8")),
                    filename=md_filename,
                ),
                caption=(
                    f"✅ <b>فایل Markdown نقشه ذهنی آماده شد!</b>\n\n"
                    f"📌 <b>موضوع مرکزی:</b> <code>{_esc(tree.text)}</code>\n"
                    f"🌿 <b>شاخه‌های اصلی ({main_branches_count}):</b> {_esc(main_titles)}\n"
                    f"🍃 <b>مجموع گره‌ها:</b> <b>{total_nodes}</b> | "
                    f"📐 <b>حداکثر عمق:</b> <b>{max_depth}</b>\n\n"
                    "💡 <i>می‌توانید این فایل را مستقیماً به ربات رندرینگ نقشه ذهنی خود فوروارد کنید.</i>"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=build_post_generation_keyboard(token),
            )
        )

        if len(canonical_md) <= 3800:
            await _safe_send(
                lambda: target_msg.reply_text(
                    text=(
                        "📋 <b>متن Markdown (با لمس کادر زیر کپی می‌شود):</b>\n\n"
                        f"<pre><code class=\"language-markdown\">{_esc(canonical_md)}</code></pre>"
                    ),
                    parse_mode=ParseMode.HTML,
                )
            )

    if status_msg:
        try:
            await status_msg.delete()
        except TelegramError:
            pass

    return WAIT_INPUT


async def handle_mode_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    if not query:
        return WAIT_INPUT

    await query.answer()
    data_str = query.data or ""
    parts = data_str.split(":", 3)
    if len(parts) != 4 or parts[0] != "GEN":
        return WAIT_INPUT

    _, token, template_mode, enrich_mode = parts
    return await _execute_ai_generation(
        update=update,
        context=context,
        token=token,
        template_mode=template_mode,
        enrich_mode=enrich_mode,
    )


async def handle_action_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    if not query:
        return WAIT_INPUT

    await query.answer()
    data_str = query.data or ""
    parts = data_str.split(":", 2)
    if len(parts) != 3 or parts[0] != "ACT":
        return WAIT_INPUT

    _, token, action = parts
    session = _get_session(context.user_data, token)
    if not session:
        if query.message:
            await query.message.reply_text(
                text="⚠️ این نشست منقضی شده است. لطفاً متن را مجدداً ارسال کنید.",
                parse_mode=ParseMode.HTML,
            )
        return WAIT_INPUT

    if action == "opml":
        tree: Optional[Node] = session.get("tree")
        if not isinstance(tree, Node):
            return WAIT_INPUT
        opml_str = tree_to_opml(tree)
        opml_filename = _safe_filename(tree.text, suffix=".opml")
        if query.message:
            await _safe_send(
                lambda: query.message.reply_document(
                    document=InputFile(
                        io.BytesIO(opml_str.encode("utf-8")),
                        filename=opml_filename,
                    ),
                    caption=(
                        f"📦 فایل استاندارد <code>OPML 2.0</code> برای <b>{_esc(tree.text)}</b>\n"
                        "قابل بازگشایی در XMind و ربات رندرینگ نقشه ذهنی."
                    ),
                    parse_mode=ParseMode.HTML,
                )
            )
        return WAIT_INPUT

    if action == "reselect":
        if query.message:
            await query.message.reply_text(
                text="🔄 لطفاً <b>الگوی ساختاری جدید</b> را برای بازسازی نقشه ذهنی انتخاب کنید:",
                parse_mode=ParseMode.HTML,
                reply_markup=build_mode_selection_keyboard(token),
            )
        return WAIT_MODE

    if action == "edit":
        context.user_data["_active_edit_token"] = token
        if query.message:
            await query.message.reply_text(
                text=(
                    "✏️ <b>حالت ویرایش هوشمند فعال شد!</b>\n\n"
                    "لطفاً تغییرات مورد نظر خود را به صورت یک پیام متنی بنویسید.\n"
                    "<i>مثال: «شاخه تشخیص افتراقی (DDx) را اضافه کن» یا «بخش درمان را به قدم اول و دوم و سوم تفکیک کن».</i>\n\n"
                    "برای انصراف از ویرایش: /cancel"
                ),
                parse_mode=ParseMode.HTML,
            )
        return WAIT_EDIT_TEXT

    return WAIT_INPUT


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        raise RuntimeError(
            "متغیر TELEGRAM_BOT_TOKEN تنظیم نشده است. لطفاً توکن ربات جدید خود را در فایل .env یا بخش Environment سایت Render وارد کنید."
        )

    keep_alive()

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .read_timeout(60.0)
        .write_timeout(60.0)
        .connect_timeout(30.0)
        .build()
    )

    doc_handler = MessageHandler(filters.Document.ALL, handle_document)
    text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    gen_cb_handler = CallbackQueryHandler(
        handle_mode_callback,
        pattern=rf"^{_CB_GEN}[0-9a-f]+:[a-z]+:[a-z]+$",
    )
    act_cb_handler = CallbackQueryHandler(
        handle_action_callback,
        pattern=rf"^{_CB_ACT}[0-9a-f]+:(?:edit|opml|reselect)$",
    )

    setkey_handler = CommandHandler("setkey", cmd_setkey)
    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            setkey_handler,
            doc_handler,
            text_handler,
            gen_cb_handler,
            act_cb_handler,
        ],
        states={
            WAIT_INPUT: [
                setkey_handler,
                doc_handler,
                text_handler,
                gen_cb_handler,
                act_cb_handler,
            ],
            WAIT_MODE: [
                setkey_handler,
                gen_cb_handler,
                act_cb_handler,
                doc_handler,
                text_handler,
            ],
            WAIT_EDIT_TEXT: [
                setkey_handler,
                text_handler,
                doc_handler,
                gen_cb_handler,
                act_cb_handler,
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel), setkey_handler],
        per_user=True,
        per_chat=True,
        allow_reentry=True,
    )

    app.add_handler(conv)
    logger.info("Starting Medical Mind Map Markdown Generator Bot polling...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
