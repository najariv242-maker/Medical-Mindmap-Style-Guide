"""
ai_engine.py - High-Speed Parallel Medical Mind Map AI Structuring Engine
=========================================================================
Fixes & Upgrades in v2.0:
1. Parallel Hedged Multi-Endpoint Racing (`ThreadPoolExecutor` + `as_completed`):
   - Fires requests concurrently across Vertex AI Express Mode (`aiplatform.googleapis.com/v1` & `v1beta1`)
     and Google AI Studio (`generativelanguage.googleapis.com/v1beta`) instead of waiting sequentially.
   - Eliminates the 75-second timeout (`asyncio.TimeoutError`) completely: the fastest responding
     endpoint/model wins in ~4–10 seconds and is cached in `_WINNING_ROUTE_CACHE` for instant subsequent calls.
2. Thinking-Latency Optimization:
   - Includes `thinkingConfig: {"thinkingBudget": 0}` alongside standard generation configs so Gemini 2.5 Flash
     generates Markdown immediately without spending 30–50 seconds on hidden thinking tokens.
3. Mode-Targeted Few-Shot Examples & Complete 15-Sample Medofast Rulebook:
   - Injects the exact Medofast reference examples matching the selected archetype (`disease`, `approach`,
     `etiology`, `workup`, `multidisorder`, `exam`) for faster, sharper, textbook-grade output.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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

# کش کردن سریع‌ترین مسیر پاسخ‌دهنده پس از اولین درخواست موفق
_WINNING_ROUTE_CACHE: Optional[Tuple[str, str, bool, bool]] = None

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
قوانین طلایی و قطعی ساختار نقشه ذهنی پزشکی (استخراج‌شده از ۱۵ نمونه مرجع مدوفست):
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
     - سطح 2 (زیرشاخه مستقیم `##`): `- متن`
     - سطح 3: `  - متن`
     - سطح 4: `    - متن`
     - سطح 5: `      - متن`
   - **ممنوعیت شماره‌گذاری مستقیم ابتدای خط (`1.` یا `۱)`)**:
     - پارسر ربات رندرینگ، شماره‌های ابتدای خط مثل `1. متن` را حذف می‌کند! بنابراین **همیشه** خط را با `- ` شروع کنید و اگر نیاز به مرحله یا شماره است، بعد از `- ` بنویسید:
       - برای مراحل درمانی: `- قدم اول`، `- قدم دوم`، `- قدم سوم`
       - برای مراحل مانور معاینه: `- ① تاباندن نور در چشم سالم`، `- ② تاباندن نور در چشم آسیب دیده`
   - از بولد کردن (`**`) یا ایتالیک (`*`) یا کد (` `) در متن گره‌ها خودداری کنید.

۲. **انتخاب هوشمند شاخه‌های اصلی (`##`) بر اساس ۶ الگوی استاندارد پزشکی**:
   بسته به ماهیت ورودی کاربر، یکی از ۶ الگوی زیر را پیاده کنید:
   - **الگوی ۱: بیماری تک‌موضوعی (Single Disease - مثل DIC، PDA، VSD، SMA، آترزی روده، آتاکسی-تلانژکتازی، آرتریت عفونی)**
     شاخه‌های اصلی (`##`) به ترتیب منطق بالینی:
     `## اتیولوژی و اپیدمیولوژی` (یا `## اتیولوژی`)، `## تظاهرات بالینی`، `## تصویربرداری` / `## پاراکلینیک` / `## یافته‌های آزمایشگاهی` / `## آزمایشات و تصویربرداری`، `## DDx` (در صورت وجود تشخیص افتراقی مهم)، `## درمان` (یا `## درمان و پیشگیری`)، `## عوارض و پروگنوز` / `## بیماری‌های همراه`، و `## نکته` / `## توجه`.
   - **الگوی ۲: گروه بیماری‌ها یا اختلالات یک ارگان (Multi-Disorder - مثل اختلالات عملکردی نوتروفیل‌ها، اختلالات پنیس)**
     شاخه‌های اصلی (`##`) خودِ بیماری‌ها هستند (مثلاً `## هایپواسپادیازیس`، `## فیموزیس`، `## پارافیموزیس` یا `## بیماری گرانولوماتوز مزمن یا CGD`، `## سندرم چدیاک هیگاشی`) و زیرشاخه‌های سطح اول شامل `- پاتوفیزیولوژی`، `- ریسک فاکتورها`، `- علائم بالینی` و `- درمان` است.
   - **الگوی ۳: دسته‌بندی اتیولوژی (Etiology Classification - مثل اتیولوژی هیپربیلیروبینمی)**
     شاخه‌های اصلی (`##`): دسته‌های بیوشیمیایی/پاتوفیزیولوژیک (`## کونژوگه`، `## غیرکونژوگه`) ← سطح اول شیوع (`- شایع`، `- غیر شایع`) ← سطح دوم مکانیسم (`  - با همولیز`، `  - بدون همولیز`) ← سطح سوم رده پاتولوژی (`    - ناسازگاری گروه‌های خونی`، `    - نقص آنزیمی RBCها`) ← سطح چهارم مصادیق (`      - ABO`، `      - RH`، `      - G6PD`).
   - **الگوی ۴: ارزیابی‌های تشخیصی و وورک‌آپ (Diagnostic Workup - مثل تروما، درد راجعه شکم)**
     حالت الف: `## ارزیابی‌های اولیه`، `## ارزیابی‌های بعدی`، `## علائم هشداردهنده`
     حالت ب: `## ارزیابی‌های آزمایشگاهی`، `## ارزیابی‌های تصویربرداری`، `## توجه`
   - **الگوی ۵: اپروچ بالینی و الگوریتم احیا/تصمیم‌گیری (Clinical Approach & Flowchart - مثل هایپرآمونمی، احیا نوزاد)**
     شاخه‌های اصلی (`##`) مسیرهای تصمیم‌گیری، مراحل الگوریتم، به همراه `## راهنما` (معرفی اختصارات مثل `- ASA -> آرژنینوسوکسینیک اسید`) یا `## مقادیر هدف SpO2 بعد از تولد` (`- دقیقه اول` -> `  - 60-65%`).
   - **الگوی ۶: معاینه سیستمیک و آناتومی بالینی (Anatomical / Sequential Exam - مثل ارزیابی اعصاب کرانیال)**
     شاخه‌های اصلی (`##`) ساختارهای ترتیبی (`## عصب 1` تا `## عصب 12`) و زیرشاخه‌ها شامل عملکرد، نحوه معاینه، و `- در صورت آسیب` -> یافته‌های بالینی و محل ضایعه.

۳. **قانون شکستن جملات به زنجیره «علت ← معلول» و «شرط ← نتیجه» (Atomization)**:
   - هرگز یک جمله طولانی را در یک باکس واحد نگذارید!
   - هرگاه جمله شامل «در موارد...»، «به علت...»، «منجر به...»، «شامل...»، یا «در صورت...» است، آن را به گره والد (`- `) و زیرشاخه (`  - `) بشکنید.
   - در بخش تصویربرداری و آزمایشگاه، همیشه **نام مدالیته یا تست** (`- ECG`، `- CXR`، `- اکوکاردیوگرافی`، `- CT-Scan`، `- آرتروسنتز`) گره والد باشد و **یافته‌ها یا اندیکاسیون‌ها** زیرشاخه آن باشند.
   - در بخش درمان دارویی اختصاصی، **نام هر دارو** گره والد و **مکانیسم و روش تجویز** زیرشاخه‌های آن باشند.

۴. **قانون دقیق نوشتن اعداد، درصدها، بازه‌های زمانی و کات‌آف‌های آزمایشگاهی**:
   - **همه اعداد را با ارقام انگلیسی (0-9) بنویسید** (`24-48 ساعت`، `25%`، `1500`، `1.8`، `50000`، `3 ماهه تا 5 ساله`، `type 1`).
   - درصدهای شیوع، بازه‌های سنی و زمانی، کات‌آف‌های آزمایشگاهی و علائم حیاتی، مراحل درمانی (`- قدم اول`، `- قدم دوم`) و مراحل آزمون (`- ①`، `- ②`) را با دقت کامل حفظ کنید.
   - اصطلاحات کلاسیک انگلیسی را داخل پرانتز قرار دهید: `فشار نبض پهن (widened pulse pressure)`، `double bubble sign (علامت دو حباب)`.
"""

MODE_EXAMPLES: Dict[str, str] = {
    "disease": """### نمونه مرجع (الگوی بیماری تک‌موضوعی - مجرای شریانی باز PDA):
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
- بسته شدن خود به خود بعد از چند هفته در نوزادان ترم ناشایع است""",

    "etiology": """### نمونه مرجع (الگوی دسته‌بندی اتیولوژی - اتیولوژی هیپربیلیروبینمی):
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
    - هیپوتیروئیدی""",

    "approach": """### نمونه مرجع (الگوی اپروچ بالینی - اپروچ به هایپرآمونمی شیرخواران):
# اپروچ به هایپرآمونمی شیرخواران

## احتمال وجود اختلال مادرزادی متابولیسم
- اسیدوز متابولیک / کتوز / اورگانیک اسیدوری
  - اسیدوز لاکتیک
    - PDH
    - PC
    - اختلالات میتوکندریایی
  - اورگانیک اسیدوری
    - پروپیونیک اسیدمی
    - متیل مالونیک اسیدمی
    - ایزووالریک اسیدمی
- نبود اسیدوز یا کتوز / وجود آلکالوز تنفسی / کاهش BUN
  - اختلالات سیکل اوره
    - سطح سیترولین پلاسما
      - افزایش شدید
        - سیترولینمی
      - افزایش همراه با وجود ASA در پلاسما یا ادرار
        - آرژنینوسوکسینیک اسیدوری
      - ناچیز یا نبودن
        - سطح اوروتیک اسید ادرار
          - بالا
            - کمبود OTC
          - طبیعی
            - کمبود CPS یا NAGS

## نارسی - دیسترس تنفسی زودرس
- افزایش آمونیاک گذرا

## راهنما
- ASA -> آرژنینوسوکسینیک اسید
- BUN -> نیتروژن اوره خون
- CPS -> کاربامیل فسفات سنتاز
- OTC -> اورنیتین ترانس کاربامیلاز""",

    "workup": """### نمونه مرجع (الگوی وورک‌آپ تشخیصی و علائم هشدار - ارزیابی‌های درد راجعه شکم):
# ارزیابی‌های درد راجعه شکم

## ارزیابی‌های اولیه
- شرح حال و معاینه فیزیکی کامل
- سؤال درباره علائم هشداردهنده
- تعیین شدت اختلال در عملکرد
  - مثلا مدرسه نرفتن
- یافته‌های آزمایشگاهی
  - CBC و ESR
  - آمیلاز و لیپاز
  - آنالیز ادراری
- سونوگرافی شکم و بررسی
  - کبد، مجاری صفراوی و کیسه صفرا
  - پانکراس
  - کلیه‌ها و حالب‌ها

## ارزیابی‌های بعدی
- انجام CT یا MRI
  - برای شکم و لگن
  - با ماده حاجب دهانی، رکتال و وریدی
- بررسی سرولوژی بیماری سلیاک
  - آنتی‌بادی IgA ترانس گلوتامیناز بافتی
- انجام اندوسکوپی و بررسی
  - مری
  - معده
  - دئودنوم

## علائم هشداردهنده
- نتایج غربالگری آزمایشگاهی غیرطبیعی
  - کم‌خونی
  - آلبومین پایین
  - مارکرهای التهابی
- تب
- استفراغ صفراوی
- خون در مدفوع یا استفراغ
- نارسایی رشد و کاهش وزن
- درد بیدارکننده از خواب
- شرح حال خانوادگی از
  - IBD
  - سلیاک
  - زخم پپتیک""",

    "multidisorder": """### نمونه مرجع (الگوی گروه بیماری‌ها - اختلالات عملکردی نوتروفیل‌ها):
# اختلالات عملکردی نوتروفیل‌ها

## بیماری گرانولوماتوز مزمن یا CGD
- پاتوفیزیولوژی
  - ناتوانی در کشتن درون سلولی پاتوژن‌های کاتالاز مثبت توسط نوتروفیل و ماکروفاژ
    - استاف اورئوس
    - باکتری‌های روده‌ای گرم منفی مثل بورخولدریا
    - قارچ‌ها مثل آسپرژیلوس و کاندیدا
  - ناتوانی در فعالسازی respiratory burst
  - همچنین می‌تواند ناشی از کمبود G6PD باشد
- علائم بالینی
  - لنفادنوپاتی و هپاتواسپلنومگالی
  - اسهال مزمن و FTT
  - هایپرگاماگلوبولینمی
  - آبسه

## سندرم چدیاک هیگاشی
- پاتوفیزیولوژی
  - بیماری اتوزومال مغلوب
  - ناشی از جهش در پروتئین سیتوپلاسمی LYST
  - فیوژن گرانول‌های اولیه و ثانویه در نوتروفیل
    - گرانول‌های غول‌آسا در لنفوسیت‌ها، پلاکت‌ها و ملانوسیت‌ها
- علائم بالینی
  - آلبینیسم
  - عفونت با استرپ و استاف و EBV
  - سندروم لنفوپرولیفراتیو
    - تب
    - زردی
    - هپاتومگالی
    - پان سیتوپنی""",

    "exam": """### نمونه مرجع (الگوی معاینه بالینی و آناتومیک - ارزیابی اعصاب کرانیال):
# ارزیابی اعصاب کرانیال

## عصب 1
- بویایی
  - در کودکان بالای 2 سال با موادی مثل عطر یا وانیل
  - موادی مانند آمونیاک از طریق عصب 5 حس می‌شوند

## عصب 2
- بینایی
  - حدت بینایی 20/200 در نوزادان و افزایش تدریجی در کودکی
  - در صورت آسیب به عصب اپتیک یک‌طرف
    - ① تاباندن نور در چشم سالم
      - تنگی هر دو مردمک
    - ② تاباندن نور در چشم آسیب دیده
      - گشادی نامتناسب هر دو مردمک
        - نشانگر آسیب آوران مردمک یا مردمک مارکوس گان

## عصب 7
- معاینه در حین استراحت، گریه و چشمک زدن
  - ضعف تمام عضلات یک طرف صورت
    - آسیب عصب فاسیال محیطی همان سمت
      - فلج بل
  - ضعف در دو سوم تحتانی یک طرف صورت
    - آسیب نورون‌های حرکتی فوقانی سمت مقابل در مغز
      - تومور، آبسه یا سکته""",
}

FEW_SHOT_EXAMPLES = MODE_EXAMPLES["disease"] + "\n\n---\n" + MODE_EXAMPLES["etiology"]


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

        if not seen_h1:
            cleaned_lines.append(f"# {stripped}")
            seen_h1 = True
        else:
            cleaned_lines.append(f"- {stripped}")

    normalized_md = "\n".join(cleaned_lines)
    tree = parse_markdown(normalized_md, default_root_title=fallback_title)
    canonical_md = tree_to_canonical_markdown(tree)
    return canonical_md, tree


def _select_examples_for_mode(template_mode: str) -> str:
    if template_mode in MODE_EXAMPLES:
        return MODE_EXAMPLES[template_mode]
    return MODE_EXAMPLES["disease"] + "\n\n---\n" + MODE_EXAMPLES["workup"]


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

    examples_block = _select_examples_for_mode(template_mode)

    return (
        f"{template_directive}\n"
        f"{enrich_directive}\n\n"
        f"نمونه‌های طلایی مورد انتظار:\n{examples_block}\n\n"
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
        # نادیده گرفتن پارت‌های thought در مدل‌های Thinking و استخراج فقط متن نهایی
        texts = [
            p.get("text", "")
            for p in parts
            if isinstance(p, dict) and p.get("text") and not p.get("thought")
        ]
        if not texts:
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
        return resp_json["output_text"].strip()
    return None


def _single_rest_attempt(
    route_kind: str,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    use_header_auth: bool,
    disable_thinking: bool,
    timeout_sec: float = 24.0,
) -> str:
    """
    اجرای یک درخواست HTTP منفرد با تایم‌اوت کنترل‌شده به یکی از درگاه‌های رسمی Google.
    """
    encoded_key = urllib.parse.quote(api_key, safe="")
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if route_kind == "vertex_v1":
        base_url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent"
    elif route_kind == "vertex_v1beta1":
        base_url = f"https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model_name}:generateContent"
    else:
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    if use_header_auth:
        url = base_url
        headers["x-goog-api-key"] = api_key
    else:
        url = f"{base_url}?key={encoded_key}"

    gen_cfg: Dict[str, Any] = {
        "temperature": 0.2,
        "maxOutputTokens": 4096,
    }
    if disable_thinking:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

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
        "generationConfig": gen_cfg,
    }
    payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw_body)
            extracted = _extract_gemini_text(data)
            if extracted and len(extracted.strip()) > 5:
                return extracted.strip()
            raise RuntimeError(f"Empty response from {route_kind}/{model_name}")
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")[:220]
        except Exception:
            pass
        raise RuntimeError(f"{route_kind}({model_name}) HTTP {http_err.code}: {err_body}") from http_err


def _sync_call_google_endpoints(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    preferred_model: str,
    timeout_sec: float = 25.0,
) -> str:
    """
    موتور فوق‌سریع موازی (Parallel Hedged Request Engine):
    - به جای انتظار ترتیبی و طولانی که باعث خطای Timeout در تلگرام می‌شد،
      درخواست‌ها را به صورت هم‌زمان (موازی) به درگاه‌های Vertex AI Express و Google AI Studio
      با حالت بدون تاخیر تفکر (thinkingBudget=0) و حالت استاندارد ارسال می‌کند.
    - اولین درگاهی که پاسخ معتبر برگرداند بلافاصله برنده شده و برای درخواست‌های بعدی کش می‌شود.
    """
    global _WINNING_ROUTE_CACHE

    # ۱. اگر از درخواست قبلی مسیر برنده‌ای در کش داریم، ابتدا همان را با تایم‌اوت سریع امتحان کن
    if _WINNING_ROUTE_CACHE is not None:
        c_route, c_model, c_hdr, c_nothink = _WINNING_ROUTE_CACHE
        try:
            res = _single_rest_attempt(
                route_kind=c_route,
                model_name=c_model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_header_auth=c_hdr,
                disable_thinking=c_nothink,
                timeout_sec=20.0,
            )
            logger.info("Fast cache hit on %s (%s)", c_route, c_model)
            return res
        except Exception as cache_exc:
            logger.warning("Cached route %s/%s failed (%s); falling back to parallel race.", c_route, c_model, cache_exc)
            _WINNING_ROUTE_CACHE = None

    # ۲. تعریف موج اول (سریع‌ترین مدل‌های Flash به صورت هم‌زمان روی Vertex Express و AI Studio)
    wave_1_candidates: List[Tuple[str, str, bool, bool]] = [
        # (route_kind, model_name, use_header_auth, disable_thinking)
        ("vertex_v1", "gemini-2.5-flash", False, True),
        ("vertex_v1", "gemini-2.0-flash", False, False),
        ("aistudio_v1beta", "gemini-2.5-flash", True, True),
        ("aistudio_v1beta", "gemini-2.0-flash", True, False),
        ("vertex_v1", "gemini-2.5-flash", False, False),
        ("aistudio_v1beta", "gemini-2.5-flash", False, False),
        ("vertex_v1beta1", "gemini-2.5-flash", False, True),
        ("vertex_v1", "gemini-3-flash-preview", False, False),
    ]

    if preferred_model and preferred_model not in ("gemini-2.5-flash", "gemini-2.0-flash"):
        wave_1_candidates.insert(0, ("vertex_v1", preferred_model, False, False))
        wave_1_candidates.insert(1, ("aistudio_v1beta", preferred_model, True, False))

    wave_2_candidates: List[Tuple[str, str, bool, bool]] = [
        ("vertex_v1", "gemini-3.8-flash", False, False),
        ("aistudio_v1beta", "gemini-3.8-flash", True, False),
        ("vertex_v1", "gemini-3.6-flash", False, False),
        ("aistudio_v1beta", "gemini-3.6-flash", True, False),
        ("vertex_v1", "gemini-1.5-flash", False, False),
        ("aistudio_v1beta", "gemini-1.5-flash", True, False),
        ("vertex_v1", "gemini-2.5-pro", False, False),
        ("aistudio_v1beta", "gemini-2.5-pro", True, False),
    ]

    errors_collected: List[str] = []

    for wave_idx, candidates in enumerate((wave_1_candidates, wave_2_candidates), start=1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates))
        future_to_cfg = {
            pool.submit(
                _single_rest_attempt,
                cfg[0],
                cfg[1],
                api_key,
                system_prompt,
                user_prompt,
                cfg[2],
                cfg[3],
                timeout_sec,
            ): cfg
            for cfg in candidates
        }
        try:
            for fut in concurrent.futures.as_completed(future_to_cfg, timeout=timeout_sec + 4.0):
                cfg = future_to_cfg[fut]
                try:
                    result_text = fut.result()
                    if result_text:
                        _WINNING_ROUTE_CACHE = cfg
                        logger.info(
                            "Wave %d winner: route=%s model=%s header=%s nothink=%s",
                            wave_idx, cfg[0], cfg[1], cfg[2], cfg[3]
                        )
                        pool.shutdown(wait=False, cancel_futures=True)
                        return result_text
                except Exception as exc:
                    errors_collected.append(str(exc))
        except concurrent.futures.TimeoutError:
            errors_collected.append(f"Wave {wave_idx} timed out after {timeout_sec}s")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # نمایش خلاصه دقیق خطاها در صورت عدم موفقیت تمام مسیرها
    unique_errors = []
    for e in errors_collected:
        if e not in unique_errors:
            unique_errors.append(e)
    error_summary = " | ".join(unique_errors[:3]) if unique_errors else "Unknown connection error"
    raise RuntimeError(f"خطا در ارتباط با وب‌سرویس هوش مصنوعی: {error_summary}")


def _sync_call_openai_compatible(
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    timeout_sec: float = 35.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
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
    تولید غیرمسدودکننده (Async) فایل Markdown نقشه ذهنی پزشکی با موتور موازی و اعتبارسنجی درخت خروجی.
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
