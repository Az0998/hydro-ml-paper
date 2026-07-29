# -*- coding: utf-8 -*-
"""一键生成《水科学进展》格式 Word 论文及全部图表。"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PAPER = ROOT / "paper"
ILLUS = PAPER / "插图（另附）"


def main():
    print("Step 1/2: 生成期刊图表与表格...")
    import generate_journal_assets
    generate_journal_assets.main()

    print("Step 2/2: 生成 Word 论文...")
    import build_word_manuscript
    build_word_manuscript.build_document()

    # 期刊要求：文稿后另附一份插图
    ILLUS.mkdir(parents=True, exist_ok=True)
    fig_src = PAPER / "figures"
    for f in sorted(fig_src.glob("*.png")):
        shutil.copy2(f, ILLUS / f.name)
    print(f"插图副本: {ILLUS}")
    print("完成！")


if __name__ == "__main__":
    main()
