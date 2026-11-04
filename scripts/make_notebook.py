#!/usr/bin/env python3
"""Create and execute the lightweight analysis notebook using committed results."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[
        md('# تحلیل جریان در شبکه تصادفی دوبعدی\n\nاین دفترچه مدل را روی یک نمونه اجرا و نتایج کامل ذخیره‌شده را بررسی می‌کند. گزارش تفصیلی فارسی در `reports/report_fa.pdf` و ارائه در `reports/presentation_en.pdf` است. برای اجرای مجدد همه ۲۴۸٬۱۸۸ ارزیابی، فرمان `make all` را از ریشه مخزن اجرا کنید.\n\nAll variables are dimensionless. One global connectivity probability is drawn per realization.'),
        code('from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd() if (Path.cwd()/"results").exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT/"src"))\nfrom flow_network import Lattice\nRES = ROOT/"results"\nFIG = ROOT/"reports"/"figures"'),
        md('## اجرای نمونه و بررسی بقای جریان\n\nجزیره‌های جدا از مرزها جریان صفر و هد مطلق نامعین دارند. حل‌گر برای رفع تکینی به آن‌ها رسانایی مصنوعی اضافه نمی‌کند.'),
        code('grid = Lattice(25)\nrng = np.random.default_rng(42)\ng = grid.generate(p=0.6, sigma=1, rng=rng)\nresult = grid.solve(g, details=True)\nassert np.isclose(result.conductance, result.outlet_flow, atol=1e-10)\nassert np.isclose(result.conductance, result.energy, atol=1e-10)\n{"G_eff":result.conductance, "spanning":result.spanning,\n "max_balance_residual":result.max_residual, "floating_nodes":result.floating_nodes}'),
        code('display(Image(filename=str(FIG/"01_network.png"), width=950))'),
        md('## آمار عدم‌قطعیت\n\n۹ سناریو، هرکدام ۱۰٬۰۰۰ نمونه مستقل. سطرهای متناظر میان سناریوها با اعداد تصادفی مشترک جفت شده‌اند. فاصله اطمینان میانگین، خطای مونت‌کارلو را نشان می‌دهد و بازه پیش‌بینی برای یک شبکه تازه نیست.'),
        code('summary = pd.read_csv(RES/"scenario_summary.csv")\nassert len(summary)==9 and (summary.samples==10000).all()\ndisplay(summary[["delta","sigma","samples","mean","mean_ci_low","mean_ci_high",\n                 "p_zero","median","q95","positive_mean"]])\ndisplay(Image(filename=str(FIG/"02_uncertainty.png"), width=950))'),
        code('samples = pd.read_csv(RES/"scenario_samples.csv.gz")\nref = samples[(samples.delta==0.25)&(samples.sigma==1)]\ng = ref.geff.to_numpy()\nassert len(g)==10000\n{"mean":g.mean(), "p_zero":float(np.mean(g==0)), "median":np.median(g),\n "q95":np.quantile(g,0.95), "conditional_positive_mean":g[g>0].mean()}'),
        code('display(Image(filename=str(FIG/"03_distribution.png"), width=950))\ndisplay(Image(filename=str(FIG/"04_connectivity.png"), width=950))'),
        md('## حساسیت گروهی سوبول\n\nپنج گروه مستقل شامل δ، σ، متغیر نهفته U برای احتمال اتصال، میدان آرایش W و میدان رسانایی Z است. رابطه `p=0.5+δU` مانع فرض استقلال اشتباه δ و p می‌شود. شاخص‌های کل هم‌پوشانی دارند و نباید به‌عنوان سهم‌های مستقل جمع شوند.'),
        code('sobol = pd.read_csv(RES/"sobol_summary.csv")\ndisplay(sobol)\ndisplay(Image(filename=str(FIG/"05_sensitivity.png"), width=1050))'),
        md('## همگرایی و کنترل میانگین رسانایی\n\nتغییر اندازه شبکه، حساسیت به اندازه جامعه شبکه‌ها را بررسی می‌کند و همگرایی یک شبکه‌بندی پیوسته نیست. توزیع اصلی میانه رسانایی را ثابت نگه می‌دارد. کنترل میانگین ثابت از مقیاس‌کردن دقیق `G*=G exp(-σ²/2)` استفاده می‌کند.'),
        code('for name in ["06_convergence", "07_normalization", "08_sobol_convergence"]:\n    display(Image(filename=str(FIG/(name+".png")), width=950))'),
        code('manifest = json.loads((RES/"manifest.json").read_text())\ndiag = pd.read_csv(RES/"solver_diagnostics.csv")\ndisplay(manifest)\ndisplay(diag[["max_residual","relative_balance_error","energy_error"]].max().to_frame("maximum"))'),
        md('## محدودیت و بازتولید\n\nمدل بی‌بُعد، دوبعدی، پایا، خطی و کالیبره‌نشده است. اعتبارسنجی عددی، اعتبار پیش‌بینی سنگ واقعی را اثبات نمی‌کند. رتبه حساسیت به توزیع‌ها و دامنه ورودی وابسته است.\n\nاز ریشه مخزن:\n```bash\npython -m pip install -r requirements.txt\npython -m pip install -e .\nmake all\n```\n\nمنابع: Jansen (1999), doi:10.1016/S0010-4655(98)00154-4؛ Kesten (1980), doi:10.1007/BF01197577. جزئیات و تفسیر کامل در گزارش‌ها آمده است.'),
    ]
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    path=ROOT/'notebooks'/'analysis.ipynb'
    path.parent.mkdir(exist_ok=True)
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)
    print(f'Executed and saved {path}')


if __name__=='__main__':main()
