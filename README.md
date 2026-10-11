# NLP 作业一

## 文件说明

| 文件 | 用途 |
|---|---|
| common.py | 数据划分、评价和逻辑回归训练 |
| task1.py | Binary 和词频实验 |
| task2.py | 三组平均词向量实验 |
| task3.py | BERT 最大长度64、微调3轮 |
| task3_prefix.py | Task 3短输入词袋对照，固定C=0.1 |
| training_size.py | 四种训练数据比例下的词频与GloVe对照 |
| analyze.py | 生成图表 |
| run.sh | 分步或完整运行 |
| requirements.txt | Python依赖 |
| data/ | 原始数据 |
| outputs/ | 结果记录 |
| report/HW1实验报告.tex | 报告源文件 |
| report/figures/ | 报告引用的全部图片 |




## 运行步骤

```bash
bash run.sh task1
bash run.sh task2
bash run.sh task3
bash run.sh prefix
bash run.sh size
bash run.sh analyze
# 或依次运行以上全部步骤：
bash run.sh all
```


