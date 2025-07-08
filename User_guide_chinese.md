# pUniFind: 统一大规模预训练深度学习模型突破质谱解析极限
这是**pUniFind**的官方仓库，目前最强大的零样本开放式肽段-谱图打分模型，性能超越现有SOTA搜索引擎，同时也是首个支持1300余种修饰的零样本开放式从头测序深度学习模型。由[pFind团队](https://pfind.net/)与[DP Technology](https://www.dp.tech/en)联合开发。
## 📚 目录
- [🚀 快速开始](#-quick-start)
- [📊 输出格式](#-output-formats)
- [📈 结果可视化](#-result-vis)
- [🔧 高级配置选项](#-configuration-options)
- [🧠 注意事项](#-take-care)
- [🛠️ 技术支持](#-technical-support)
- [❓ 常见问题](#-faq)
- [🤝 引用说明](#-citation)
## 🚀 快速开始 <a name="-quick-start"></a>
演示数据可从[Google Drive](https://drive.google.com/drive/folders/1CQzNypmOscCpvyK3MnCj4AhEWGlx9mbn?usp=sharing)下载。
### Windows本地部署
请先下载```.exe```[安装包](https://github.com/pFindStudio/pUniFind/releases)，然后按照指引安装。在Windows上运行需要GPU支持。普通用户需注册免费Bohrium账号。

关于```GPU batch size```设置：
- 显存≥8GB时建议设为128
- 显存≈4GB时建议设为64

可通过终端命令```nvidia-smi```查看显存信息。

重打分结果将保存在```result```文件夹，从头测序结果将存储在```pUniFind_result```文件夹。
### Linux本地部署（即将发布）
#### 环境配置
pUniFind支持多GPU加速处理。
| 环境 | 版本 |
| :---: | ---: |
| cuda | >= 11.7 |
| python | 3.8 |
```bash
# 创建conda环境
conda create -n pUniFind python=3.8 -y
conda activate pUniFind
# 进入项目目录
cd Contrast_MS_Pep
bash env.sh
```
#### 开放式重打分
将以下文件夹放入```official_projects```：
```bash
project_name/ # pFind任务文件夹
├── param/ # pFind搜索参数（由pFind生成）
├── result/ # pFind搜索结果（由pFind生成）
│   └── ***.pac # 蛋白质ID文件（需用户手动移动到fasta文件夹）
└── mgfs/ # mgf文件（由pFind生成后用户移动至此）
```
执行重打分：
```bash
bash official_score_workflow.sh project_name batchsize
```
建议初始批量大小设为256，根据运行速度和显存调整。
- 结果文件保存为```project_namefdr0.01_pUniFind.spectra```
#### 开放式从头测序
将以下文件夹放入```official_projects```：
```bash
project_name/ 
└── mgfs/ # mgf文件（由pFind生成后用户移动至此）   
```
执行从头测序：
```bash
bash official_denovo_workflow.sh project_name batchsize
```
建议初始批量大小设为256，根据运行速度和显存调整。
- 直接测序结果保存为```pUniFind_result```文件夹下的```project_name_001_5_merged.csv```和```project_name_001_5_filtered.csv```
- 修饰统计信息保存为```pUniFind_result```文件夹下的```project_name_mod.txt```
- 所有连接肽段保存为```pUniFind_result```文件夹下的```project_name.fasta```
若仅关注**少量修饰**，建议使用上述fasta文件通过pFind3（关闭开放模式）进行搜索，并参考```project_name_mod.txt```设置可变修饰。
### 网页应用
无GPU资源时可通过[Bohrium网页端](https://bohrium.dp.tech/apps/punifind)租用GPU在线运行。

Bohrium的GPU资源可能存在波动，若任务无法启动通常由资源不足导致。建议优先选择4090显卡，其次3090。
遇到问题请通过**技术支持**联系我们。
```bash
# 需上传的文件结构
# 重打分
pFind任务文件夹/ # 由pFind生成
├── param/ # pFind搜索参数
├── result/ # pFind搜索结果
│   └── ***.pac # 蛋白质ID文件
└── mgfs/ # mgf文件
# 从头测序
项目文件夹/ 
└── mgfs/ # mgf文件   
```
## 📊 输出格式 <a name="-output-formats"></a>
### 重打分结果
| 列名 | 含义 | 示例 |
| :---: | :--- | --- |
| File_Name | mgf文件中的谱图标题 | example.1.1.2.0.dta |
| Scan_No | 扫描号 | 1000 |
| Charge | 电荷数 | 1 |
| Sequence | pUniFind识别的序列 | SPTCTNQEL |
| Calc_MHplus+ | 理论MH+质量 | 2031.948724 |
| Modification | 修饰信息 | 4,Carbamidomethyl[C];8,Cation_Na[E]; |
| Proteins | 关联蛋白 | tr\|A0A075B6G3\|A0A075B6G3_HUMAN/ |
### 从头测序结果
目前为提升性能，仅预测母离子质量误差在20ppm内且长度6-40的肽段。超出范围的预测结果将不被记录。未来版本将支持更灵活设置。

为便于展示格式，将列名转为行展示：
#### 对于_merged.csv和_filtered.csv
| 列名 | 含义 | 示例 |
| :---: | :--- | --- |
| spectrum title | mgf文件中的谱图标题 | example.1.1.2.0.dta |
| score | pUniFind预测分数（与重打分分数一致） | 7.241 |
| cos similarity | 实验谱图与预测肽段谱图余弦相似度 | 0.95 |
| Retention time | 实验保留时间（秒） | 1169.002807 |
| Missing fragment ion site | 缺失碎片离子位置（最后一个下划线后数字为肽段长度，需忽略） | 6_8 |
| mass difference | 预测肽段与实验前体质量差 | 0.0003662109375 |
| Peptide sequence | 预测肽段序列 | ['SPTCTNQEL'] |
| Peptide sequence with modification | 带修饰位点的预测序列 | SPTCTNQEL_4_Carbamidomethyl_8_Cation_Na |
| Modifications | 预测修饰及位点 | "4,Carbamidomethyl[C];8,Cation_Na[E];" |
#### 对于.fasta
标准FASTA格式文件。
#### 对于_mod.txt
| 列名 | 含义 | 示例 |
| :---: | :--- | --- |
| Modification Name | 修饰名称 | Oxidation[M] |
| Frequency of modification | 在前N候选中出现频率 | 3296 |
## 🔧 高级配置选项 <a name="-configuration-options"></a>
可在shell脚本（official_denovo_workflow.sh, official_score_workflow.sh）中修改以下配置：
| 参数名 | 用途 |
| :---: | --- |
| num_proc | 数据处理时的CPU进程数（当mgf文件较多时有用，num_proc <= CPU核心数，默认16） |
| range_pred | 预测不同长度候选肽段的数量（先预测长度再预测序列，需为奇数，默认5） |
<!-- | fdr_thread(仅重打分) | pFind谱图FDR阈值（低于该值的谱图将被重新打分，默认0.1） | -->
未来将支持更多配置：
| 参数名 | 用途 |
| :---: | --- |
| de novo min/max length | 预测肽段的最小/最大长度（先预测长度再预测序列，不符合长度要求的谱图将跳过以加速） |
| predict_score_all | 当所有候选肽段都不满足20ppm阈值时是否仍预测分数（默认跳过以加速） |
| instrument | 仪器类型（如QE, Lumos, TIMS, Astral，默认QE） |
| nce file path | 碎裂能量文件路径（默认30） |
## 📈 结果可视化 <a name="-result-vis"></a>
我们为论文中提到的两种工作流提供可视化工具：
- 常规从头测序：使用[pLabel](https://pfind.ict.ac.cn/se/plabel/)可视化谱图。需要.plabel文件和对应的.mgf文件。用户需修改.plabel文件中的mgf路径（由pUniFind生成）。pLabel使用指南见链接。**务必确保mgf文件名和路径正确！**
```bash
# pLabel格式示例
[FilePath]
File_Path=C:\Users\Ecoli-E1-F2-20151208_HCDFT_extract103.mgf # mgf文件路径
[Modification]
1=Oxidation[M]
2=Carbamidomethyl[C]
[xlink]
xlink=NULL
[Total]
total=1
[Spectrum1]
name=ECOLI-E1-F2-20151208.30360.30360.3.0.DTA
pep1=0 LGLDVLVHGEAER 1 
```
- 修饰丰富型从头测序：依赖[pFind](https://pfind.net/se/pFind/index.html)（关闭开放模式）搜索pUniFind生成的数据库，结果可通过pBuild可视化（已集成在pFind中）。
## 🧠 注意事项 <a name="-take-care"></a>
数据类型：
- 目前不支持ITMS（分辨率较低的过时模式）或ETD/EThcD数据。对于Astral 窄窗口DIA从头测序，我们建议用户先使用timsTOF模式进行从头测序。Astral 窄窗口DIA数据比较稀少，如果用户可以贡献Astral 窄窗口DIA数据，我们愿意提供finetune服务让pUniFind在Astral上效果更优。

开放从头测序极具挑战性，需注意：
- 存在若干"质量巧合"修饰：
```Q+Deamidated[Q]=E```, ```N+Deamidated[N]=D```, ```glycidamide[任意]=S```, ```Acetyl+K=AV/VA```,```K+Crotonyl=PV/VP```,```K+Formy=GV/VG```,```K+Ubiq=GG```,```G+Methyl=A```等。除非特别需要，否则不建议在修饰丰富型工作流中搜索这些修饰（可通过后处理过滤）。具体修饰信息可参考pFind安装目录或GitHub仓库的```modification.ini```文件。
- 可忽略的丢失修饰：
```Arg-loss[AnyC-termR]```, ```Met-loss[ProteinN-termM]```, ```Met-loss+Acetyl[ProteinN-termM]```等。
## 🛠️ 技术支持  <a name="-technical-support"></a>
如遇技术问题、性能异常或发现结果与评估指标不一致，欢迎反馈🙏。我们正在收集异常案例以改进模型。**正在**持续更新优化软件，因主要开发者**毕业还远** :(。

**技术问题：**  
1. **GitHub Issues**: [新建Issue](https://github.com/pFindStudio/pUniFind/issues)需包含：  
   - 数据描述  
   - 错误日志和环境信息  
   - 上传文件说明  
1. **pFind Studio用户支持微信群**：  
   - 添加微信：```JL_Zhao2000```，将邀请入群（微信邀请链接有效期为一周）。  

**合作咨询：**  
📧 **联系人**: 赵家乐 邮箱: [zhaojiale22z@ict.ac.cn](mailto:zhaojiale22z@ict.ac.cn) 或 [marshmallowzjl@gmail.com](mailto:marshmallowzjl@gmail.com)。
## ❓ 常见问题 <a name="-faq"></a>
- **MGF格式**: 请确保使用pFind生成的MGF文件。不同软件生成的MGF格式差异较大。最新版pFind支持Thermo、timsTOF等仪器数据。重打分/从头测序时，可直接用pFind搜索.raw/.d文件生成MGF（需在MS Data的Data Extraction中点击`MGF`）。若坚持使用MSConvert生成的MGF，可把mgf一起放到某个文件夹下，通过脚本处理：  
  ```bash
  python3 mgf_processor.py -i /mgf输入路径/ -o /处理后mgf输出路径/ -p 核数(默认8)
  ```
- **安装路径**: 安装路径和数据/结果路径请勿包含空格/中文。  
- **Windows卸载**: 如需重新安装Windows版，请使用```unins000.exe```卸载，否则可能无法更改安装路径。若已通过其他方式卸载，请重新安装后使用上述方法卸载。
- **Linux部署**: 如遇```libstdc++.so.6: version `GLIBCXX_3.4.29' not found```错误，参考[此方案](https://github.com/pybind/pybind11/discussions/3453)。我的解决方法是```export LD_LIBRARY_PATH=/your_path/miniconda3/envs/pUniFind/lib:$LD_LIBRARY_PATH```。
- **结果缺失**: 若未获得重打分结果，请检查.pac文件位置，同时查看```pUniFind_result```和```result```文件夹。
- **靶向方法**: 使用靶向采集方法（如AIMS、PRM、SRM）时性能不理想，请联系团队。我们可分析数据并推荐优化策略。
## 🤝 引用说明 <a name="-citation"></a>
如本软件对您的研究有帮助，请**引用**我们🙏：
```bash
@misc{zhao2025punifindunifiedlargepretrained,
      title={pUniFind: a unified large pre-trained deep learning model pushing the limit of mass spectra interpretation}, 
      author={Jiale Zhao and Pengzhi Mao and Kaifei Wang and Yiming Li and Yaping Peng and Ranfei Chen and Shuqi Lu and Xiaohong Ji and Jiaxiang Ding and Xin Zhang and Yucheng Liao and Weinan E and Weijie Zhang and Han Wen and Hao Chi},
      year={2025},
      eprint={2507.00087},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.00087}, 
}
```
您的每次引用都将激励开发者改进pUniFind的易用性和功能。毕业生需要您的引用和GitHub Star收藏来获得工作机会 😫。