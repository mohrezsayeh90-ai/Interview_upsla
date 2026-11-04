#!/usr/bin/env python3
"""Create and execute a compact notebook using trained weights and saved tests."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]


def main():
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[
        md('# PINN پارامتری گرافی و مقایسه با حل مستقیم\n\nاین دفترچه وزن آموزش‌دیده را بارگذاری و بدون آموزش دوباره برای ورودی تازه استفاده می‌کند. گزارش فارسی در `parametric_pinn/reports/comparison_fa.pdf` است.\n\nمعماری، یک شبکه گرافی با تعداد ثابت گام به‌روزرسانی فیزیکی هد است؛ پیش‌پردازش مؤلفه و فاصله مقاومتی نیز برای هر ورودی اجرا می‌شود. این مدل، یک MLP اسکالر یا فرمول بسته نیست. برای یک شبکه مشخص، میدان رسانایی یا بذر آن لازم است؛ پارامترهای آماری به‌تنهایی جواب یکتا نمی‌دهند.'),
        code('from pathlib import Path\nimport sys,json,time\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nsys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))\nfrom parametric_pinn.model import Predictor\nB=ROOT/"parametric_pinn"\nR=B/"results"\nF=B/"reports"/"figures"\nmanifest=json.loads((R/"training_manifest.json").read_text())\npredictor=Predictor(B/"models/parametric_pinn.pt",depth=manifest["accurate_depth"])\n{"trainable_parameters":manifest["parameters"],"depth":predictor.depth,\n "training_reference_labels":manifest["training_reference_labels"]}'),
        md('## استفاده از وزن‌ها برای ورودی تازه\n\nورودی شامل رسانایی تک‌تک لینک‌ها، δ، p، σ و هد مرزهاست. مقیاس رسانایی و اختلاف هد با روابط فیزیکی دقیق اعمال می‌شوند. تابع `predict` هیچ حل مستقیم مرجعی اجرا نمی‌کند.'),
        code('grid=predictor.preprocessor.grid\nrng=np.random.default_rng(42)\ng=grid.generate(p=.6,sigma=1,rng=rng)\nt=time.perf_counter()\npred=predictor.predict(g,delta=.25,p=.6,sigma=1,head_left=1,head_right=0)\nresult={k:v for k,v in pred.items() if k!="heads"}\nresult["seconds_warm_call"]=time.perf_counter()-t\nresult'),
        md('حل مرجع زیر فقط برای بررسی در دفترچه است و بخشی از مسیر پیش‌بینی مدل نیست. رسانایی انرژی و شار ورودیِ هد تقریبی باید جدا بررسی شوند.'),
        code('reference=grid.solve(g,details=True)\n{"G_reference":reference.conductance,"G_PINN_energy":pred["geff"],\n "relative_error":pred["geff"]/reference.conductance-1 if reference.conductance else 0,\n "inlet_flux_PINN":pred["inlet_flow_from_heads"],\n "outlet_flux_PINN":pred["outlet_flow_from_heads"],\n "reference_balance_residual":reference.max_residual}'),
        md('## پیش‌بینی دسته‌ای و آمار\n\nاین مثال کوچک فقط استفاده از API را نشان می‌دهد؛ برآورد دقیق توزیع به نمونه‌های بیشتری نیاز دارد. شبکه‌های غیرعبوری با قاعده دقیق گراف، جریان صفر دارند.'),
        code('rng=np.random.default_rng(6789)\nps=.5+.25*rng.uniform(-1,1,32)\nfields=np.stack([grid.generate(p,1,rng) for p in ps])\nvalues=predictor.predict_batch(fields,delta=.25,p=ps,sigma=1)\n{"draws":len(values),"mean_G":values.mean(),"p_zero":np.mean(values==0),\n "median":np.median(values),"q95":np.quantile(values,.95)}'),
        md('## آموزش فیزیکی بدون برچسب جواب\n\n۴۰۹۶ نمونه عبوری برای آموزش و ۵۱۲ نمونه مستقل برای اعتبارسنجی استفاده شدند. عمق و وزن براساس اعتبارسنجی انتخاب شدند و سپس آزمون مستقل اجرا شد.'),
        code('display(pd.read_csv(R/"training_history.csv"))\ndisplay(pd.read_csv(R/"validation_depths.csv"))\ndisplay(Image(filename=str(F/"01_training.png"),width=1000))'),
        md('## مقایسه روی میدان یکسان\n\nخطاهای نسبی فقط برای موارد مرجع مثبت‌اند. احتمال جریان صفر جدا محاسبه می‌شود. کنترل بدون آموزش، سهم لایه‌های فیزیکی را از بهبود ناشی از یادگیری جدا می‌کند.'),
        code('summary=pd.read_csv(R/"test_summary.csv")\ndisplay(summary[["split","method","cases","positive_cases","positive_mean_abs_relative_error",\n "positive_p95_abs_relative_error","positive_max_abs_relative_error","mean_free_node_rmse"]])\ndisplay(Image(filename=str(F/"02_accuracy.png"),width=1000))\ndisplay(Image(filename=str(F/"03_head_fields.png"),width=1000))'),
        code('display(pd.read_csv(R/"scenario_comparison.csv").query("method == \'pinn_accurate\'"))\ndisplay(Image(filename=str(F/"04_uncertainty_comparison.png"),width=1000))\ndisplay(pd.read_csv(R/"sobol_comparison.csv").query("method == \'pinn_accurate\' and order == \'total\'"))\ndisplay(Image(filename=str(F/"05_sensitivity_comparison.png"),width=1000))'),
        md('## هزینه، تعمیم و محدودیت\n\nروی این شبکه کوچک و CPU، مدل عصبی سریع‌تر از حل مستقیم نبود. دقت رسانایی مبتنی بر انرژی، بقای دقیق شار یا دقت همه هدهای موضعی را تضمین نمی‌کند. هدف صدک ۹۹ خطای اعتبارسنجی کمتر از ۰٫۱٪ در سقف عمق آزموده‌شده محقق نشد و در گزارش ثبت شده است.'),
        code('timing=pd.read_csv(R/"timing_benchmark.csv")\ndisplay(timing.groupby(["method","batch_size"])[["milliseconds_per_case","speedup_vs_sparse_g"]].median())\ndisplay(Image(filename=str(F/"06_cost_accuracy.png"),width=1000))\ndisplay(Image(filename=str(F/"07_generalization_physics.png"),width=1000))'),
        md('## بازتولید\n\nاز ریشه مخزن، پس از نصب وابستگی‌ها:\n```bash\nmake -C parametric_pinn all\n```\nبرای استفاده از وزن موجود، نیاز به اجرای آموزش نیست:\n```bash\npython parametric_pinn/scripts/predict.py --delta .25 --sigma 1 --p .6 --seed 42\n```\nمنابع و تمام روش‌ها در گزارش فارسی و انگلیسی آمده‌اند. مطالعه اولیه حل مستقیم در پوشه‌های اصلی مخزن حفظ شده است.'),
    ]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=ROOT/'parametric_pinn/notebooks/comparison.ipynb';path.parent.mkdir(exist_ok=True)
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)
    print(f'Executed {path}')


if __name__=='__main__':main()
