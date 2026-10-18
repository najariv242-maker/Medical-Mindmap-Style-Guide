"""
ai_engine.py - Medical Mind Map AI Structuring Engine & Multi-Endpoint Gemini/Vertex Client
============================================================================================
- Supports Google AI Studio Auth Keys & Vertex AI Express Keys starting with `AQ.` (e.g. AQ.Ab8RN6...)
  as well as standard `AIza...` keys and OpenAI-compatible keys without false regex rejections.
- Implements the complete Medofast Medical Mind Map Rulebook derived from 15 clinical reference maps.
- Post-processes and validates every output through `parser.py` to guarantee 100% compatibility
  with `mindmapbotargon-1.0.0`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from parser import Node, parse_markdown, tree_to_canonical_markdown

logger = logging.getLogger("mindmap_md_bot.ai_engine")

DEFAULT_AI_API_KEY = "AQ.Ab8RN6Ijxd6MDZtSc4gAuQlpeJR8VXhRAt2EF3CTkR5f66Qo4Q"

TEMPLATE_MODES: Dict[str, Dict[str, str]] = {
    "auto": {
        "label": "🤖 تشخیص خودکار الگو (پیشنهادی)",
        "desc": "انتخاب هوشمند بهترین ساختار بر اساس نوع مبحث پزشکی (بیماری، اپروچ، اتیولوژی، وورک‌آپ یا معاینه)",
    },
    "disease": {
        "label": "🦠 بیماری تک‌موضوعی (مثل DIC, PDA, SMA)",
        "desc": "شاخه‌های اصلی: اتیولوژی و اپیدمیولوژی، تظاهرات بالینی، پاراکلینیک/تصویربرداری، DDx، درمان، عوارض و نکات",
    },
    "approach": {
        "label": "🧭 اپروچ و الگوریتم بالینی (مثل احیا و هایپرآمونمی)",
        "desc": "ساختار مرحله‌به‌مرحله، نقاط تصمیم‌گیری (بله/خیر یا سطوح آزمایشگاهی)، مقادیر هدف و راهنمای اختصارات",
    },
    "etiology": {
        "label": "🗂 دسته‌بندی اتیولوژی / DDx (مثل هیپربیلیروبینمی)",
        "desc": "تفکیک پاتوفیزیولوژیک (مثل کونژوگه/غیرکونژوگه) ← شیوع (شایع/غیرشایع) ← مکانیسم (با همولیز/بدون همولیز) ← علل",
    },
    "workup": {
        "label": "🔎 ارزیابی تشخیصی و علائم خطر (مثل تروما و درد شکم)",
        "desc": "شاخه‌های اصلی: ارزیابی‌های اولیه/آزمایشگاهی، ارزیابی‌های بعدی/تصویربرداری، علائم هشداردهنده (Red Flags) و توجه",
    },
    "multidisorder": {
        "label": "📚 گروه بیماری‌ها / چند اختلال (مثل اختلالات نوتروفیل و پنیس)",
        "desc": "هر بیماری به عنوان یک شاخه اصلی (##) و زیرشاخه‌های پاتوفیزیولوژی، علائم بالینی و درمان در دل هر کدام",
    },
    "exam": {
        "label": "🧠 معاینه و آناتومی بالینی (مثل اعصاب کرانیال)",
        "desc": "تفکیک ساختاری/شماره‌ای (مثلاً عصب 1 تا 12) همراه با عملکرد، نحوه معاینه، مراحل آزمون (①، ②) و یافته‌های آسیب",
    },
}

ENRICH_MODES: Dict[str, Dict[str, str]] = {
    "enrich": {
        "label": "✨ تکمیل علمی هوشمند (بر اساس رفرنس پزشکی)",
        "desc": "تبدیل حتی یک جمله ساده یا نام بیماری به نقشه ذهنی کامل و استاندارد بر اساس رفرنس‌های پزشکی (نلسون/هاریسون)",
    },
    "strict": {
        "label": "🔒 وفادار به متن ارسالی (بدون افزودن مطالب خارج از متن)",
        "desc": "ساختاربندی و بهینه‌سازی دقیق همان مطالب ارسالی شما بدون اضافه کردن اطلاعات خارجی",
    },
}

MEDICAL_MINDMAP_SYSTEM_PROMPT = """شما یک متخصص برجسته آموزش پزشکی و طراح حرفه‌ای نقشه‌های ذهنی (Medical Mind Map Architect) به سبک استاندارد «مدوفست (Medofast)» هستید.
وظیفه شما دریافت متن‌های بسیار ساده، جملات پراکنده، جزوات خام، یا حتی فقط نام یک مبحث پزشکی از کاربر و تبدیل آن به **بهترین، دقیق‌ترین و استانداردترین فایل Markdown** برای ربات رندرینگ نقشه ذهنی (XMind / Argon Renderer) است.

=====================================================================
قوانین طلایی و قطعی ساختار نقشه ذهنی پزشکی (استخراج‌شده از نمونه‌های مرجع):
=====================================================================

۱. **قانون سینتکس خروجی (صددرصد سازگار با Parser ربات آرگون)**:
   - خروجی باید **فقط و فقط کد خام Markdown** باشد (بدون هیچ توضیح اضافه قبل یا بعد از آن).
   - **موضوع مرکزی (Central Topic - سطح 0)**: دقیقاً یک خط در ابتدای فایل با `# ` (یک هشتگ).
     - اگر مبحث یک بیماری دارای اختصار رایج است، نام فارسی + اختصار انگلیسی در پرانتز بیاید:
       مثال: `# انعقاد داخل عروقی منتشر (DIC)`، `# مجرای شریانی باز (PDA)`، `# نقص دیواره بین بطنی (VSD)`، `# آتروفی عضلانی نخاعی (SMA)`، `# آرتریت عفونی (چرکی یا سپتیک)`
     - اگر مبحث اپروچ، اتیولوژی یا ارزیابی است، دقیقاً با همان عنوان کلیدی آغاز شود:
       مثال: `# اتیولوژی هیپربیلیروبینمی`، `# اپروچ به هایپرآمونمی شیرخواران`، `# ارزیابی‌های تشخیصی در تروما`، `# ارزیابی‌های درد راجعه شکم`، `# ارزیابی اعصاب کرانیال`
   - **شاخه‌های اصلی (Main Topics - سطح 1)**: فقط با `## ` (دو هشتگ). هرگز از `###` یا `####` استفاده نکنید!
   - **زیرشاخه‌ها (Subtopics - سطح 2 به بعد)**: فقط با `- ` (خط تیره و یک فاصله) و **تورفتگی دقیق ۲ فاصله (2 spaces) به ازای هر سطح عمق**:
     - سطح 2 (زیرشاخه مستقیم `##`): بدون فاصله قبل از `- ` (یعنی `- متن`)
     - سطح 3: ۲ فاصله قبل از `- ` (یعنی `  - متن`)
     - سطح 4: ۴ فاصله قبل از `- ` (یعنی `    - متن`)
     - سطح 5: ۶ فاصله قبل از `- ` (یعنی `      - متن`)
   - **ممنوعیت شماره‌گذاری مستقیم ابتدای خط (`1.` یا `۱)`)**:
     - پارسر ربات رندرینگ، شماره‌های ابتدای خط مثل `1. متن` را حذف می‌کند! بنابراین **همیشه** خط را با `- ` شروع کنید و اگر نیاز به مرحله یا شماره است، بعد از `- ` بنویسید:
       - برای مراحل درمانی: `- قدم اول`، `- قدم دوم`، `- قدم سوم`
       - برای مراحل مانور معاینه: `- ① تاباندن نور در چشم سالم`، `- ② تاباندن نور در چشم آسیب دیده`
   - از بولد کردن (`**`) یا ایتالیک (`*`) یا کد (` `) در متن گره‌ها خودداری کنید (چون ربات رندرینگ وزن فونت هر سطح را خودش تنظیم می‌کند).

۲. **انتخاب هوشمند شاخه‌های اصلی (`##`) بر اساس ۶ الگوی استاندارد پزشکی**:
   بسته به ماهیت ورودی کاربر، یکی از ۶ الگوی زیر را پیاده کنید:

   - **الگوی ۱: بیماری تک‌موضوعی (Single Disease - مثل DIC، PDA، VSD، SMA، آترزی روده، آتاکسی-تلانژکتازی، آرتریت عفونی)**
     شاخه‌های اصلی (`##`) به ترتیب منطق بالینی:
     - `## اتیولوژی و اپیدمیولوژی` (یا `## اتیولوژی` / `## پاتوفیزیولوژی`)
     - `## تظاهرات بالینی`
     - `## تصویربرداری` یا `## پاراکلینیک` یا `## یافته‌های آزمایشگاهی` یا `## آزمایشات و تصویربرداری` (متناسب با بیماری)
     - `## DDx` (در صورت اهمیت تشخیص افتراقی مثل مقایسه DIC با نارسایی کبدی، کمبود ویتامین K و سپسیس)
     - `## درمان` (یا `## درمان و پیشگیری`)
     - `## عوارض و پروگنوز` یا `## بیماری‌های همراه` (در صورت وجود در مبحث)
     - `## نکته` یا `## توجه` (برای نکات طلایی، سن شروع خاص، یا هشدارهای کلیدی بالینی در انتهای نقشه)

   - **الگوی ۲: گروه بیماری‌ها یا اختلالات یک ارگان (Multi-Disorder - مثل اختلالات عملکردی نوتروفیل‌ها، اختلالات پنیس)**
     - شاخه‌های اصلی (`##`) خودِ بیماری‌ها یا زیرگونه‌ها هستند:
       مثال ۱: `## بیماری گرانولوماتوز مزمن یا CGD` و `## سندرم چدیاک هیگاشی`
       مثال ۲: `## هایپواسپادیازیس`، `## فیموزیس`، `## پارافیموزیس`
     - زیرشاخه‌های سطح اول (`- `) در دل هر بیماری شامل: `- پاتوفیزیولوژی`، `- ریسک فاکتورها`، `- علائم بالینی`، `- درمان`.

   - **الگوی ۳: دسته‌بندی اتیولوژی و تشخیص افتراقی (Etiology Classification - مثل اتیولوژی هیپربیلیروبینمی)**
     - شاخه‌های اصلی (`##`): دسته‌های اصلی بیوشیمیایی/پاتوفیزیولوژیک (مثلاً `## کونژوگه` و `## غیرکونژوگه`).
     - زیرشاخه سطح اول (`- `): تفکیک بر اساس شیوع (`- شایع` و `- غیر شایع`).
     - زیرشاخه سطح دوم (`  - `): تفکیک بر اساس مکانیسم (`  - با همولیز` و `  - بدون همولیز`).
     - زیرشاخه سطح سوم (`    - `): دسته علت (`    - ناسازگاری گروه‌های خونی`، `    - نقص آنزیمی RBCها`، `    - اختلالات غشا RBCها`).
     - زیرشاخه سطح چهارم (`      - `): مصادیق دقیق (`      - ABO`، `      - RH`، `      - G6PD`، `      - پیرووات کیناز`).

   - **الگوی ۴: ارزیابی‌های تشخیصی و وورک‌آپ (Diagnostic Workup - مثل تروما، درد راجعه شکم)**
     - حالت الف (بر اساس مرحله و علائم خطر - مثل درد راجعه شکم):
       `## ارزیابی‌های اولیه`، `## ارزیابی‌های بعدی`، `## علائم هشداردهنده`
     - حالت ب (بر اساس نوع پاراکلینیک - مثل تروما):
       `## ارزیابی‌های آزمایشگاهی`، `## ارزیابی‌های تصویربرداری`، `## توجه`

   - **الگوی ۵: اپروچ بالینی و الگوریتم احیا/تصمیم‌گیری (Clinical Approach & Flowchart - مثل هایپرآمونمی، احیا نوزاد)**
     - شاخه‌های اصلی (`##`) مسیرهای اصلی تشخیصی، مراحل الگوریتم، به همراه `## راهنما` (معرفی اختصارات مثل `- ASA -> آرژنینوسوکسینیک اسید`) یا `## مقادیر هدف SpO2 بعد از تولد` (`- دقیقه اول` -> `  - 60-65%`).
     - در اپروچ‌های آزمایشگاهی، هر تست یک گره والد است و وضعیت‌های آن (`- افزایش شدید`، `- طبیعی یا پایین`، `- بالا`، `- ناچیز یا نبودن`) زیرشاخه‌های آن هستند که به تشخیص نهایی می‌رسند.

   - **الگوی ۶: معاینه سیستمیک و آناتومی بالینی (Anatomical / Sequential Exam - مثل ارزیابی اعصاب کرانیال)**
     - شاخه‌های اصلی (`##`) ساختارهای ترتیبی هستند (`## عصب 1` تا `## عصب 12`).
     - زیرشاخه‌ها شامل عملکرد (`- بویایی`، `- بینایی`، `- حرکات چشم`)، نحوه معاینه، و `- در صورت آسیب` -> یافته‌های بالینی و محل ضایعه (مثلاً تفکیک آسیب محیطی/فلج بل از آسیب نورون حرکتی فوقانی).

۳. **قانون شکستن جملات به زنجیره «علت ← معلول» و «شرط ← نتیجه» (Atomization)**:
   - هرگز یک جمله طولانی را در یک باکس واحد نگذارید!
   - هرگاه جمله شامل «در موارد...»، «به علت...»، «منجر به...»، «شامل...»، یا «در صورت...» است، آن را به گره والد و زیرشاخه بشکنید:
     *غلط*: `- در موارد متوسط تا بزرگ نارسایی قلبی و ادم ریوی رخ می‌دهد که با تجویز مایعات وریدی تشدید می‌شود`
     *صحیح (دقیقاً مطابق نمونه)*:
     ```markdown
     - موارد متوسط تا بزرگ
       - نارسایی قلبی و ادم ریوی
         - تشدید با تجویز مایعات وریدی
         - رال و هپاتومگالی
     ```
   - در بخش تصویربرداری و آزمایشگاه، همیشه **نام مدالیته یا تست** گره والد باشد و **یافته‌ها یا شرایط انجام آن** زیرشاخه آن باشند:
     ```markdown
     - ECG
       - بزرگی و هایپرتروفی دهلیز و بطن چپ
     - CXR
       - کاردیومگالی، بزرگی بطن چپ، افزایش سیلوئت شدن شریان پولمونری، افزایش جریان خون پولمونری
       - احتمال بزرگی و هایپرتروفی بطن راست به خاطر هایپرتنشن پولمونری
     - اکوکاردیوگرافی
       - بررسی محل و اندازه ضایعه
     ```
   - در بخش درمان دارویی اختصاصی (مثل SMA)، **نام هر دارو** گره والد و **مکانیسم و روش تجویز** زیرشاخه‌های آن باشند:
     ```markdown
     - nusinersen
       - از دسته الیگونوکلئوتیدهای آنتی سنس
       - تزریق‌های متناوب داخل نخاعی (اینتراتکال)
     - onasemnogene abeparvovec
       - یک درمان ژنی
       - یک نوبت تزریق داخل وریدی
     - risdiplam
       - یک مولکول کوچک
       - مصرف روزانه به شکل خوراکی
     ```

۴. **قانون دقیق نوشتن اعداد، درصدها، بازه‌های زمانی و کات‌آف‌های آزمایشگاهی**:
   - **همه اعداد را با ارقام انگلیسی (0-9) بنویسید** (مثل `24-48 ساعت`، `25%`، `1500`، `1.8`، `50000`، `3 ماهه تا 5 ساله`، `type 1`) تا در رندرینگ BiDi کنار علامت `%`، `-`، `>` و `<` کاملاً تمیز و بدون به‌هم‌ریختگی نمایش داده شوند.
   - **کجا حتماً باید عدد بنویسیم؟**
     1. **درصد شیوع و رتبه‌بندی اپیدمیولوژیک**: `حدود 25% بیماری‌های congenital قلبی`، `زانوها (40%)`، `لگن (20%)`، `مچ پا (14%)`، `یک مورد در هر 1500 تولد زنده`، `حدود 55% بیماران فرم شدید اوایل شیرخوارگی را دارند`.
     2. **بازه‌های سنی و زمانی بالینی**: `24-48 ساعت بعد از تولد`، `ظهور در روز 2-4`، `طی 6-8 هفته`، `سن زیر 2 ماه`، `بالای 2 ماه تا بلوغ`، `کودکان 3 ماهه تا 5 ساله`، `شروع در 2 تا 5 سالگی`، `قبل از 18 ماهگی`، `حداقل 21 روز`.
     3. **کات‌آف‌های آزمایشگاهی، علائم حیاتی و کنتراندیکاسیون‌ها**: `کراتینین بالای 1.8`، `پلاکت زیر 50000`، `ضربان قلب کمتر از 100؟`، `ضربان قلب کمتر از 60؟`، `GCS ≤ 14`، `شفاف - 0-200 سلول - زیر 10% نوتروفیل`، `کدر - 10000-250000 سلول - بالای 90% نوتروفیل`.
     4. **مراحل درمانی ترتیبی**: `- قدم اول`، `- قدم دوم`، `- قدم سوم`.
     5. **مراحل آزمون بالینی دو یا چند مرحله‌ای**: `- ① ...`، `- ② ...`.
     6. **تایپ‌ها، کروموزوم‌ها و فاکتورها**: `type 1/Werdnig-Hoffmann disease`، `روی کروموزوم 11 (11q22.3)`، `فاکتورهای 2، 5 و 8`.

۵. **قانون اصطلاحات انگلیسی و پرانتزها (سازگاری با موتور BiDi رندرر)**:
   - اصطلاحات کلاسیک پزشکی یا معادل‌های انگلیسی مهم را داخل پرانتز قرار دهید تا موتور BiDi رندرر آن‌ها را در یک بلوک یکپارچه قرار دهد:
     مثال: `فشار نبض پهن (widened pulse pressure)`، `سوفل ماشینی مداوم (continuous machine-like murmur)`، `کلاپس شدن قفسه سینه (bell-shaped chest)`، `double bubble sign (علامت دو حباب)`.
   - طول متن هر گره باید کوتاه و اسکن‌پذیر باشد (ترجیحاً ۲ تا ۱۰ کلمه؛ حداکثر ۷۰ کاراکتر).
"""

FEW_SHOT_EXAMPLES = """
### نمونه ۱ (الگوی بیماری تک‌موضوعی - مجرای شریانی باز PDA):
# مجرای شریانی باز (PDA)

## اتیولوژی و اپیدمیولوژی
- مجرای شریانی موجب حرکت خون از شریان ریوی به آئورت طی زندگی جنینی می‌شود
- به طور طبیعی در نوزادان ترم 24-48 ساعت بعد از تولد بسته می‌شود
- از عوارض سندروم دیسترس تنفسی در نوزادان
  - در نوزادان پره ترم با RDS بسته نمی‌شود
    - شانت چپ به راست و افزایش جریان خون ریوی
  - فاز حاد RDS
    - هایپوکسی، هایپرکاپنه و اسیدوز منجر به وازوکانستریکشن شریان پولمونری می‌شود

## تظاهرات بالینی
- ظهور در روز 2-4
- موارد کوچک
  - بی‌علامت
- موارد متوسط تا بزرگ
  - نارسایی قلبی و ادم ریوی
    - تشدید با تجویز مایعات وریدی
    - رال و هپاتومگالی
- فشار نبض پهن (widened pulse pressure)
  - نبض محیطی قابل لمس و جهنده
- گاهی سوفل ماشینی مداوم (continuous machine-like murmur)
  - زیر کلاویکل چپ
  - البته معمولا جزء سیستولیک آن سمع می‌شود
- hyperdynamic precordium

## تصویربرداری
- موارد کوچک
  - نوار قلب و گرافی نرمال
- موارد متوسط تا بزرگ
  - سیلوئت شدن کامل شریان پولمونری
  - کاردیومگالی و ادم ریوی در گرافی سینه
  - افزایش واسکولاریته ریوی
  - هایپرتروفی بطن چپ
  - هایپرتروفی بطن راست در صورت هایپرتنشن ریوی
- اکوگرافی و داپلر
  - بررسی اندازه و آناتومی PDA
  - سایز قلب چپ و فشار شریان ریوی

## درمان
- موارد کوچک و بی‌علامت ممکن است درمان نخواهند
- قدم اول
  - ممنوعیت مایعات + دیورتیک
- قدم دوم
  - در صورت بسته نشدن تا 24-48 ساعت
    - مهارکننده‌های پروستاگلاندین سنتتاز، ایبوپروفن، ایندومتاسین یا استامینوفن
- قدم سوم
  - حدود 20-30% نوزادان به درمان فوق پاسخ نمی‌دهند
    - جراحی
      - coil embolization
      - closure device
- کنتراندیکاسیون‌های مصرف ایندومتاسین
  - خونریزی
  - اولیگوری
  - کراتینین بالای 1.8
  - پلاکت زیر 50000
- بسته شدن خود به خود بعد از چند هفته در نوزادان ترم ناشایع است

---
### نمونه ۲ (الگوی دسته‌بندی اتیولوژی - اتیولوژی هیپربیلیروبینمی):
# اتیولوژی هیپربیلیروبینمی

## کونژوگه
- شایع
  - کلستاز هیپرالیمانتاسیون
  - عفونت CMV
  - عفونت‌های مادرزادی پری‌ناتال
    - TORCH
  - تغلیظ صفراوی به علت همولیز طولانی
  - هپاتیت نوزادی
  - عفونت‌های ادراری
  - سپسیس
- غیر شایع
  - انفارکتوس کبد
  - اختلالات ارثی متابولیسم
    - گالاکتوزمی
    - تیروزینمی
  - کمبود آلفا-یک آنتی‌تریپسین
  - فیبروز کیستیک
  - آترزی مجاری صفراوی
  - کیست کلدوک
  - هموکروماتوز نوزادی
  - سندرم Alagille
  - بیماری Byler
  - کلستاز داخل کبدی فامیلیال پیشرونده تایپ 1و2و3
  - pan hypopituitarism

## غیرکونژوگه
- شایع
  - با همولیز
    - ناسازگاری گروه‌های خونی
      - ABO
      - RH
      - Kell
      - Duffy
    - عفونت
  - بدون همولیز
    - یرقان فیزیولوژیک
    - یرقان شیر مادر
    - خونریزی داخلی
    - پلی‌سیتمی
    - نوزادان مادر دیابتی
- غیر شایع
  - با همولیز
    - نقص آنزیمی RBCها
      - G6PD
      - پیرووات کیناز
    - اختلالات غشا RBCها
      - اسفروسیتوز
      - اووالوسیتوز
    - هموگلوبینوپاتی
      - تالاسمی
  - بدون همولیز
    - موتاسیون آنزیم گلوکورنیل ترانسفراز
      - کریگلر نجار
      - ژیلبرت
    - تنگی پیلور
    - هیپوتیروئیدی
    - ترومبوسیتوپنی ایمونولوژیک
"""


_PERSIAN_DIGITS_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBERED_LINE_RE = re.compile(r"^([ \t]*)(?:[-*+•]\s+)?(\d+|[۰-۹]+)[.\)\-]\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)


def normalize_digits_to_english(text: str) -> str:
    """تبدیل ارقام فارسی/عربی به انگلیسی برای جلوگیری از به‌هم‌ریختگی BiDi در کنار علائم ریاضی و %."""
    return text.translate(_PERSIAN_DIGITS_MAP)


def postprocess_and_validate_markdown(raw_ai_output: str, fallback_title: str = "نقشه ذهنی پزشکی") -> Tuple[str, Node]:
    """
    پاک‌سازی، استانداردسازی تورفتگی‌ها، حفظ اعداد مرحله‌ای در پارسر و اعتبارسنجی نهایی با parser.py
    """
    text = (raw_ai_output or "").strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    else:
        fence_find = re.search(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if fence_find:
            text = fence_find.group(1).strip()
        else:
            first_h1 = re.search(r"(?m)^#\s+.+$", text)
            if first_h1:
                text = text[first_h1.start():].strip()

    text = normalize_digits_to_english(text)
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cleaned_lines: List[str] = []
    seen_h1 = False
    circled_nums = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if re.match(r"^([-*_])\1{2,}$", stripped):
            continue

        # تبدیل ### یا بیشتر به زیرشاخه بولت‌دار تا ساختار فقط # و ## و - باشد
        hm = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if hm:
            hashes, content = hm.groups()
            content = content.strip()
            if len(hashes) == 1:
                if not seen_h1:
                    cleaned_lines.append(f"# {content}")
                    seen_h1 = True
                else:
                    cleaned_lines.append(f"## {content}")
            elif len(hashes) == 2:
                cleaned_lines.append(f"## {content}")
            else:
                indent = "  " * (len(hashes) - 3)
                cleaned_lines.append(f"{indent}- {content}")
            continue

        # محافظت از شماره‌های مرحله‌ای در برابر حذف شدن توسط _BULLET_RE در parser.py
        nm = _NUMBERED_LINE_RE.match(line)
        if nm:
            indent_ws, num_str, rest = nm.groups()
            indent_cols = len(indent_ws.expandtabs(2))
            indent_prefix = "  " * (indent_cols // 2)
            try:
                n_val = int(num_str)
                if 1 <= n_val <= len(circled_nums):
                    badge = circled_nums[n_val - 1]
                else:
                    badge = f"مرحله {n_val}:"
            except ValueError:
                badge = f"{num_str}-"
            cleaned_lines.append(f"{indent_prefix}- {badge} {rest.strip()}")
            continue

        bm = re.match(r"^([ \t]*)[-*+•]\s+(.+)$", line)
        if bm:
            indent_ws, content = bm.groups()
            indent_cols = len(indent_ws.expandtabs(2))
            indent_prefix = "  " * (indent_cols // 2)
            cleaned_lines.append(f"{indent_prefix}- {content.strip()}")
            continue

        # خط بدون علامت
        if not seen_h1:
            cleaned_lines.append(f"# {stripped}")
            seen_h1 = True
        else:
            cleaned_lines.append(f"- {stripped}")

    normalized_md = "\n".join(cleaned_lines)
    tree = parse_markdown(normalized_md, default_root_title=fallback_title)
    canonical_md = tree_to_canonical_markdown(tree)
    return canonical_md, tree


def build_user_prompt(
    raw_input: str,
    template_mode: str = "auto",
    enrich_mode: str = "enrich",
    edit_instruction: Optional[str] = None,
    previous_markdown: Optional[str] = None,
) -> str:
    mode_info = TEMPLATE_MODES.get(template_mode, TEMPLATE_MODES["auto"])

    if edit_instruction and previous_markdown:
        return (
            f"نقشه ذهنی Markdown فعلی:\n```markdown\n{previous_markdown.strip()}\n```\n\n"
            f"دستور ویرایش کاربر:\n«{edit_instruction.strip()}»\n\n"
            "لطفاً نقشه ذهنی بالا را دقیقاً بر اساس دستور کاربر و با رعایت کامل قوانین سبک مدوفست "
            "به‌روزرسانی کن و فقط کد کامل Markdown نهایی را برگردان."
        )

    template_directive = (
        f"الگوی ساختاری انتخاب‌شده: {mode_info['label']} — {mode_info['desc']}"
        if template_mode != "auto"
        else "الگوی ساختاری: به صورت هوشمند بهترین الگو را از بین ۶ الگوی استاندارد مدوفست (بیماری تک‌موضوعی، گروه بیماری‌ها، دسته‌بندی اتیولوژی، وورک‌آپ تشخیصی، اپروچ بالینی، یا معاینه آناتومیک) متناسب با متن انتخاب کن."
    )

    if enrich_mode == "enrich":
        enrich_directive = (
            "حالت تکمیل علمی (Smart Medical Enrichment): فعال.\n"
            "اگر ورودی کاربر کوتاه، ناقص، یک جمله ساده یا صرفاً نام یک بیماری/مبحث است، "
            "تمام شاخه‌های استاندارد و نکات کلیدی، درصدهای شیوع، بازه‌های سنی، کات‌آف‌های آزمایشگاهی، "
            "علائم بالینی، یافته‌های تصویربرداری/آزمایشگاهی و مراحل درمان (قدم اول، قدم دوم...) را "
            "دقیقاً بر اساس رفرنس‌های معتبر پزشکی (نلسون، هاریسون و آپتودیت) و به سبک نمونه‌های مدوفست تکمیل کن."
        )
    else:
        enrich_directive = (
            "حالت وفاداری مطلق به متن (Strict Mode): فعال.\n"
            "فقط و فقط مطالب موجود در متن کاربر را دسته‌بندی، خرد (Atomize) و به شاخه‌ها و زیرشاخه‌های استاندارد تبدیل کن "
            "و مطلبی خارج از متن ورودی اضافه نکن."
        )

    return (
        f"{template_directive}\n"
        f"{enrich_directive}\n\n"
        f"نمونه‌های طلایی مورد انتظار:\n{FEW_SHOT_EXAMPLES}\n\n"
        f"=====================================================================\n"
        f"متن یا موضوع ورودی کاربر:\n«{raw_input.strip()}»\n"
        f"=====================================================================\n"
        f"اکنون خروجی نهایی را فقط به صورت کد استاندارد Markdown (شروع با `# عنوان مرکزی`، شاخه‌های اصلی با `## ` و زیرشاخه‌ها با `- ` و تورفتگی ۲ فاصله) تولید کن:"
    )


def _extract_gemini_text(resp_json: Dict[str, Any]) -> Optional[str]:
    candidates = resp_json.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
        return resp_json["output_text"].strip()
    return None


def _sync_call_google_endpoints(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    preferred_model: str,
    timeout_sec: float = 55.0,
) -> str:
    """
    پشتیبانی کامل و بدون خطا از کلیدهای جدید Google AI Studio / Vertex AI Express با پیشوند `AQ.`
    و همچنین کلیدهای `AIza...`:
    - تلاش خودکار روی هر دو درگاه رسمی Google:
      1. Vertex AI Express Mode (`aiplatform.googleapis.com`) با پارامتر `?key=`
      2. Google AI Studio Native (`generativelanguage.googleapis.com`) با هدر `x-goog-api-key` و پارامتر `?key=`
      3. کتابخانه رسمی `google-genai` در صورت نصب بودن
    """
    models_to_try: List[str] = []
    for m in [
        preferred_model,
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-3-flash-preview",
        "gemini-1.5-pro",
    ]:
        m_clean = (m or "").strip()
        if m_clean and m_clean not in models_to_try:
            models_to_try.append(m_clean)

    payload_dict = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 8192,
        },
    }
    payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
    encoded_key = urllib.parse.quote(api_key, safe="")

    last_errors: List[str] = []

    # ۱. تلاش با SDK رسمی google-genai در صورت نصب بودن
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        for vertex_flag in (True, False) if api_key.startswith("AQ.") else (False, True):
            try:
                client = genai.Client(api_key=api_key, vertexai=vertex_flag)
                for model_name in models_to_try[:2]:
                    try:
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.25,
                                max_output_tokens=8192,
                            ),
                        )
                        if resp and getattr(resp, "text", None):
                            return resp.text.strip()
                    except Exception as sdk_model_exc:
                        last_errors.append(f"SDK(vertex={vertex_flag},{model_name}): {sdk_model_exc}")
            except Exception as sdk_init_exc:
                last_errors.append(f"SDKInit(vertex={vertex_flag}): {sdk_init_exc}")
    except ImportError:
        pass

    # ۲. تلاش مستقیم روی REST APIهای رسمی Vertex AI Express و Google AI Studio
    is_aq_key = api_key.startswith("AQ.")

    for model_name in models_to_try:
        endpoints: List[Tuple[str, str, Dict[str, str]]] = [
            (
                "VertexAI-Express-v1",
                f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent?key={encoded_key}",
                {"Content-Type": "application/json"},
            ),
            (
                "VertexAI-Express-v1beta1",
                f"https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model_name}:generateContent?key={encoded_key}",
                {"Content-Type": "application/json"},
            ),
            (
                "AIStudio-v1beta-Header",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                {"Content-Type": "application/json", "x-goog-api-key": api_key},
            ),
            (
                "AIStudio-v1beta-Query",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={encoded_key}",
                {"Content-Type": "application/json"},
            ),
        ]

        if not is_aq_key:
            endpoints = endpoints[2:] + endpoints[:2]

        for route_label, url, headers in endpoints:
            req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                    raw_body = response.read().decode("utf-8", errors="replace")
                    data = json.loads(raw_body)
                    extracted = _extract_gemini_text(data)
                    if extracted:
                        logger.info("Successfully generated mind map via %s (%s)", route_label, model_name)
                        return extracted
            except urllib.error.HTTPError as http_err:
                err_body = ""
                try:
                    err_body = http_err.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last_errors.append(f"{route_label}({model_name}) HTTP {http_err.code}: {err_body}")
            except Exception as exc:
                last_errors.append(f"{route_label}({model_name}): {exc}")

    error_summary = " | ".join(last_errors[:4]) if last_errors else "Unknown API error"
    raise RuntimeError(f"خطا در ارتباط با وب‌سرویس هوش مصنوعی: {error_summary}")


def _sync_call_openai_compatible(
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    timeout_sec: float = 55.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.25,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
        return data["choices"][0]["message"]["content"].strip()


async def generate_medical_mindmap_markdown(
    raw_input: str,
    template_mode: str = "auto",
    enrich_mode: str = "enrich",
    edit_instruction: Optional[str] = None,
    previous_markdown: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Node]:
    """
    تولید غیرمسدودکننده (Async) فایل Markdown نقشه ذهنی پزشکی و اعتبارسنجی درخت خروجی.
    """
    active_key = (
        api_key
        or os.getenv("AI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or DEFAULT_AI_API_KEY
    ).strip()

    preferred_model = (os.getenv("AI_MODEL") or "gemini-2.5-flash").strip()
    custom_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    user_prompt = build_user_prompt(
        raw_input=raw_input,
        template_mode=template_mode,
        enrich_mode=enrich_mode,
        edit_instruction=edit_instruction,
        previous_markdown=previous_markdown,
    )

    if custom_base_url and not active_key.startswith(("AQ.", "AIza")):
        raw_md = await asyncio.to_thread(
            _sync_call_openai_compatible,
            active_key,
            custom_base_url,
            MEDICAL_MINDMAP_SYSTEM_PROMPT,
            user_prompt,
            preferred_model,
        )
    else:
        raw_md = await asyncio.to_thread(
            _sync_call_google_endpoints,
            active_key,
            MEDICAL_MINDMAP_SYSTEM_PROMPT,
            user_prompt,
            preferred_model,
        )

    canonical_md, tree = postprocess_and_validate_markdown(raw_md)
    return canonical_md, tree
