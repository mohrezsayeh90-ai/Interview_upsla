# PINN پارامتری گرافی و مقایسه با حل مستقیم

این پوشه، مطالعه مستقل روش عصبی را نگه می‌دارد. خروجی‌های مطالعه اولیه با حل‌گر تنک در پوشه‌های اصلی مخزن حفظ شده‌اند.

- [گزارش کامل فارسی، ۹ صفحه](reports/comparison_fa.pdf) · [HTML فارسی](reports/comparison_fa.html)
- [English report, 9 pages](reports/comparison_en.pdf)
- [ارائه مقایسه‌ای، ۱۲ اسلاید](reports/comparison_slides.pdf) · [PowerPoint قابل‌ویرایش](reports/comparison_slides.pptx)
- [وزن آموزش‌دیده](models/parametric_pinn.pt) · [دفترچه اجراشده](notebooks/comparison.ipynb)
- [نتایج خام و تعریف ستون‌ها](results/README.md)

## مدل چه کاری انجام می‌دهد؟

مدل یک نگاشت پارامتری گرافی است: کل رسانایی لینک‌ها و پارامترهای δ، p و σ را می‌گیرد، هد را پیش‌بینی و رسانایی مؤثر را از انرژی تخمین می‌زند. وزن‌ها یک بار آموزش دیده‌اند و برای شبکه‌های تصادفی دیده‌نشده دوباره استفاده می‌شوند.

**معماری از لایه‌های فیزیکی با گام‌های بازشده استفاده می‌کند.** هر پیش‌بینی، پیش‌پردازش مؤلفه‌های گراف و فاصله مقاومتی و تعداد ثابت K گام به‌روزرسانی هد را اجرا می‌کند. در زمان استفاده، وزن‌ها دوباره آموزش داده نمی‌شوند و دستگاه تنک مرجع حل نمی‌شود. این مدل، فرمول بسته یا صرفاً یک MLP با ورودی چند اسکالر نیست.

برای یک تحقق مشخص، δ، p و σ کافی نیستند: آرایش لینک‌ها و رسانایی تک‌تک آن‌ها نیز لازم‌اند. API این میدان را می‌گیرد؛ CLI می‌تواند با یک بذر تصادفی آن را بسازد. اگر فقط پارامترهای توزیع را بدهید، خروجی آماری با نمونه‌برداری میدان‌ها محاسبه می‌شود.

## نتایج اندازه‌گیری‌شده

| معیار | PINN، عمق ۲۵۶ | PINN، عمق ۲۰۴۸ |
|---|---:|---:|
| میانگین خطای نسبی مطلق رسانایی در شبکه‌های مثبت آزمون سناریویی | ۰٫۴۷۰۵٪ | **۰٫۰۲۰۰٪** |
| صدک ۹۵ همان خطا | ۲٫۹۰۰۵٪ | **۰٫۰۳۹۳٪** |
| بیشترین همان خطا | ۱۸٫۸۳۵۵٪ | **۴٫۷۳۸۶٪** |
| میانگین RMSE هد گره‌های آزاد | ۰٫۰۰۹۹۰۹ | **۰٫۰۰۱۴۶۰** |
| زمان دسته‌ای B=64، میلی‌ثانیه برای هر تحقق | ۲٫۵۰۸ | ۱۵٫۰۹۷ |

آزمون سناریویی ۹٬۰۰۰ میدان جدید دارد؛ ۴٬۵۱۸ مورد دارای جریان هستند. علاوه بر آن، ۱۰۰۰ آزمون پیوسته و چهار خانواده خارج دامنه، هرکدام ۲۵۰ نمونه، اجرا شدند. در σ=۳ خارج از دامنه آموزش، میانگین خطای نسبی حالت دقیق حدود **۲٫۰۸٪** و بیشترین آن **۲۰٫۲۶٪** شد.

مدل در شبکه‌های بدون مسیر، جریان دقیقاً صفر می‌دهد؛ این نتیجه از تحلیل مؤلفه‌ها می‌آید و یک موفقیت طبقه‌بندی یادگیری نیست. هر **۳۶ آزمون خودکار** موفق است: ۲۴ آزمون حل‌گر اولیه و ۱۲ آزمون جدید PINN.

**مقایسه زمان:** حل مستقیم فقط رسانایی، حدود ۰٫۶ میلی‌ثانیه برای هر نمونه زمان برد. PINN در این شبکه کوچک و روی CPU سریع‌تر نبود؛ هزینه آموزش و بارگذاری نیز باید جدا در نظر گرفته شوند. داده‌های خام سه تکرار و هر دو نوع زمان مستقیم و عصبی در `results/timing_benchmark.csv` هستند.

## آموزش، فیزیک و دامنه اعتبار

- ۴۰۹۶ میدان عبوری برای آموزش، ۵۱۲ میدان عبوری مستقل برای اعتبارسنجی؛ بذر پایه `20261105` و جریان‌های تصادفی جدا.
- **هیچ جواب مرجع هد یا رسانایی، برچسب آموزش نیست.** آموزش، انرژی دیریکله و باقیمانده کیرشهف را کمینه می‌کند. جواب‌های مستقیم برای انتخاب مدل با اعتبارسنجی و ارزیابی بعدی استفاده شدند.
- دو MLP کوچک، تصحیح هد اولیه و ضرایب لایه‌های فیزیکی را یاد می‌گیرند؛ جمعاً ۱۹۳۹ پارامتر آموزش‌پذیر.
- ۱۶ دوره آموزش با عمق ۳۲، ۶۴، ۱۲۸ و ۲۵۶؛ دوره ۱۵ با اعتبارسنجی انتخاب شد. حلقه آموزش حدود ۲۹۷ ثانیه و کل آماده‌سازی و انتخاب عمق حدود ۳۳۵ ثانیه طول کشید.
- آموزش روی CPU با float32 و ارزیابی با float64 و دو رشته PyTorch بود. نتایج GPU یا برتری زمانی برای شبکه‌های بزرگ‌تر ادعا نشده است.
- دامنه آموزش: n=25، δ در [۰،۰٫۲۵]، p نمونه‌برداری‌شده پیرامون ۰٫۵، و σ در [۰،۲]. انتقال اندازه به n=15,41 صرفاً آزمایش خارج دامنه است.
- هدف صدک ۹۹ خطای اعتبارسنجی کمتر از ۰٫۱٪ تا سقف عمق ۲۰۴۸ محقق نشد؛ مقدار مشاهده‌شده حدود ۰٫۳۷٪ بود. حالت دقیق، دقیق‌ترین عمق آزموده‌شده است و تضمین خطای یکنواخت نیست.

## رسانایی انرژی و شار را جدا ببینید

برای هد واحد و شرط مرزی دقیق:

```text
E(h;g) = Σ_edges g (h_u - h_v)^2
∂E/∂h_i = 2 Σ_j g_ij (h_i - h_j)
G_exact = min_h E(h;g)
G_PINN = E(h_PINN;g) ≥ G_exact       (در حساب دقیق)
```

این تخمین تا وقتی هد کاملاً به تعادل نرسیده، با شار ورودی محاسبه‌شده از هد تقریبی برابر نیست. در حالت دقیق آزمون سناریویی، صدک ۹۵ خطای نسبی شار ورودی حدود **۱٫۱۲٪** و صدک ۹۵ عدم‌تطابق نسبی ورودی و خروجی حدود **۳٫۰۲٪** بود. همه این کمیت‌ها گزارش شده‌اند؛ دقت G به‌تنهایی به‌معنی بقای دقیق جریان یا دقت همه هدهای موضعی نیست.

بدنه محاسبه، یک مدل گرافی فیزیک‌آگاه/تکراریِ آموزش‌پذیر است؛ دقت نهایی عمدتاً از لایه‌های فیزیکی می‌آید. در عمق ثابت ۲۵۶، آموزش نسبت به کنترل بدون آموزش، میانگین خطای رسانایی را حدود ۲٫۴۲ برابر بهبود داد. مقایسه کنترل بدون آموزش در همان فایل‌های خام موجود است.

## نصب و استفاده از وزن موجود

از ریشه مخزن، با Python 3.12:

```bash
python -m pip install -r parametric_pinn/requirements.txt
python -m pip install -e .
python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --p .6 --seed 42
```

نسخه ثبت‌شده PyTorch برابر `2.15.0+cpu` است. برای نصب جداگانه نسخه CPU:

```bash
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**در فرمان پیش‌فرض، حل مستقیم فراخوانی نمی‌شود.** برای مقایسه اختیاری همان ورودی و ذخیره هد:

```bash
python parametric_pinn/scripts/predict.py --p .6 --sigma 1 --seed 42 \
  --compare-reference --save-heads work/predicted_heads.npy
```

برای ورودی رسانایی مشخص، یک فایل NumPy با شکل `(1200,)` بدهید؛ ترتیب لینک‌ها همان مدل اولیه است: افقی و سپس عمودی، هرکدام سطرمحور.

```bash
python parametric_pinn/scripts/predict.py --conductances field.npy --p .6 --sigma 1
```

اختلاف هد و ضریب مقیاس رسانایی هم قابل‌تغییرند:

```bash
python parametric_pinn/scripts/predict.py --p .6 --sigma 1 --scale 2 \
  --head-left 3 --head-right 1
```

هد از تبدیل خطی دقیق و رسانایی از مقیاس‌پذیری دقیق استفاده می‌کند. تغییر اندازه شبکه در API پیش‌فرض n=25 پشتیبانی عمومی اعلام نشده؛ روش انتقال اندازه در کد ارزیابی موجود است.

برای تخمین آمار با پارامترهای توزیع، بدون حل مستقیم:

```bash
python parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --draws 1000 --seed 123
```

این فرمان میدان‌های تصادفی را نمونه‌برداری و با همان وزن‌ها به‌صورت دسته‌ای پیش‌بینی می‌کند. `mcse_of_surrogate_mean` خطای مونت‌کارلو است و خطای تقریب شبکه عصبی را شامل نمی‌شود.

## API در پایتون

از ریشه مخزن:

```python
import numpy as np
from flow_network import Lattice
from parametric_pinn.model import Predictor

solver = Predictor("parametric_pinn/models/parametric_pinn.pt", depth=2048)
grid = Lattice(25)
g = grid.generate(p=0.6, sigma=1, rng=np.random.default_rng(42))
result = solver.predict(g, delta=0.25, p=0.6, sigma=1,
                        head_left=1, head_right=0)
print(result["geff"], result["inlet_flow_from_heads"])
heads = result["heads"]
```

`Predictor.predict_batch` آرایه `(batch,1200)` می‌گیرد و رسانایی همه موارد را با یک وزن بارگذاری‌شده می‌دهد. هد مطلق جزیره‌های شناور با `NaN` برگردانده می‌شود. برای حالت سریع `depth=256` بگذارید.

## بازتولید کامل

```bash
make -C parametric_pinn all
```

این دستور آموزش، آزمون مستقل، حساسیت، زمان‌سنجی، شکل‌ها، گزارش‌ها، ارائه، دفترچه و ممیزی را از نو اجرا می‌کند. فقط برای بازسازی گزارش از نتایج موجود:

```bash
make -C parametric_pinn report notebook audit
```

بدون Make، از ریشه مخزن به‌ترتیب اجرا کنید:

```bash
python parametric_pinn/scripts/verify.py
OPENBLAS_NUM_THREADS=1 python parametric_pinn/scripts/train.py
OPENBLAS_NUM_THREADS=1 python parametric_pinn/scripts/evaluate.py
python parametric_pinn/scripts/make_deliverables.py
python parametric_pinn/scripts/make_notebook.py
python parametric_pinn/scripts/audit.py
```

در PowerShell، متغیر محیطی را جدا تنظیم کنید: `$env:OPENBLAS_NUM_THREADS="1"`. ساخت PDF به Pango و قلم Noto Naskh Arabic نیاز دارد؛ راهنمای نصب بسته‌های سیستمی در README اصلی آمده است. زمان مطالعه عددی مقایسه‌ای روی این میزبان حدود ۴۶۹ ثانیه، علاوه بر آموزش بود؛ زمان و فراداده PDF/دفترچه با محیط اجرا تغییر می‌کنند.

## مقایسه کامل دو روش

مقایسه جفت‌شده شامل خطای رسانایی، هد، شار لینک‌ها و بقای جریان؛ میانگین، انحراف معیار، میانه و صدک ۹۵ و احتمال صفر؛ حساسیت گروهی سوبول روی ۲۰۴۸ جفت پایه با ۱۰۰۰ بوت‌استرپ؛ آزمون پیوسته و خارج دامنه؛ هزینه آموزش و پیش‌پردازش و پیش‌بینی تک‌نمونه و دسته‌ای است.

نتایج مطالعه اولیه با ۱۰٬۰۰۰ نمونه در هر سناریو و نمونه تازه با ۱۰۰۰ مورد در هر سناریو نباید مستقیم از هم کم شوند و «خطای شبکه» نام بگیرند. ستون‌های جفت‌شده، خطای مدل را روی ورودی دقیقاً یکسان جدا می‌کنند. تمام محدودیت‌های مدل بی‌بُعد و شبکه منظم دوبعدیِ کالیبره‌نشده همچنان برقرارند.

## English summary

This is a separate **unrolled parametric graph PINN / learned iterative operator**. It uses an energy/residual loss with no training solution labels, a geodesic encoder and learned residual-layer coefficients. Each new field still requires graph preprocessing and a fixed number of iterative head updates. There is no sparse reference solve or per-input weight fitting in the deployment path.

The complete conductance field is required to identify a realization. δ,p,σ alone define a distribution, not a unique network response. At K=2048, positive-case mean absolute relative conductance error is 0.0200% on 9,000 held-out scenario cases, with p95 0.0393% and maximum 4.7386%. Inlet flux and local head errors are separately reported. The CPU model is slower than the sparse solver on this small problem. See the English PDF for full methods, ablations, uncertainty, sensitivity and limitations.

References: Raissi et al. (2019), doi:10.1016/j.jcp.2018.10.045; E and Yu (2018), doi:10.1007/s40304-018-0127-z; Jansen (1999), doi:10.1016/S0010-4655(98)00154-4.
