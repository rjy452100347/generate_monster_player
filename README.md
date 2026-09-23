# MapleStoryClassicDatasetGenerator

从 `mxdclassic` 客户端的 Unity Addressables、WZJS 和 WZSS 资源离线合成
`1280×224` 检测数据。当前综合流程支持 `0: monster`、`1: player` 二分类，
并保留早期单类 Pilot／dense 流程。生成器不会修改客户端，也不会把 GMS v83 素材混入主数据。

本仓库仅包含生成、校验与复现代码，不包含游戏客户端、客户端素材、缓存或已生成的数据集。
使用前请安装 Python 3.11，并在所选 YAML 配置中填写本机的 `client_root`、`cache_dir` 和
`output_root`：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**人物／怪物二分类请先阅读 [使用与调用说明](docs/人物怪物数据集使用与调用说明.md)**，
包含资产路径、环境准备、命令行与 Python 调用、续传条件、48K 扩展及训练配置说明。

2026-09-20 核对：已有 60K、48K 和 97 张稀有怪物补丁。两套主数据的历史校验报告仍有覆盖问题；
当前代码生成的调度哈希与旧 60K 不同，原一键脚本不能直接续传旧 60K。请使用独立输出名称试运行，
不要对保留数据使用 `--overwrite`。当前客户端目录的资源 catalog 为 1.15.2，旧缓存目录名不代表版本锁定。

以下保留早期各阶段的使用记录；其中“单类 monster”和 Pilot 验收结果不代表当前二分类数据的状态。

## 使用

```powershell
cd 'D:\workspace\冒险岛脚本测试\开发参考\代码上库\历史项目\MapleStoryClassicDatasetGenerator'
.\.venv\Scripts\python.exe -m classic_dataset.generate --config configs\pilot.yml
.\.venv\Scripts\python.exe -m classic_dataset.validate --dataset 'F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\pilot'
```

先用 `--limit 12 --overwrite` 运行烟雾集。Pilot 通过验收后，再使用 `configs\formal.yml` 生成正式集。

输出同时包含 COCO 主清单、YOLO images/labels、`data.yaml`、地图拆分、资源索引、统计、缺失报告、
失败地图报告和前 100 张可视化框检查图。train/val/test 按地图 ID 拆分。

首次运行会将两个大 UnityFS 图集流式解压到 `F:\MapleStoryAssets\.classic_cache\unity_bundles`；
这是为避免 UnityPy 将 8 GB 以上内容拼接到内存。缓存可复用，但不要在生成期间删除。

## 当前 Pilot

已生成到 `F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\pilot`：2,000 张，
共 2,325 个怪物框，负样本比例 10.85%，train/val/test 使用 144/18/18 张互不重叠的地图。
`validation_report.json` 为零错误、零警告。正式 20,000 张应在人工查看 `visual_checks` 后再启动。

## 密集多实例增强集

生成器现在支持每图随机 `1–20` 个怪物框，并提供 `dense` 采样：怪物仍位于真实出生点对应的平台，
密集图按精灵宽度自适应间距，制造局部遮挡和框重叠，而不旋转或缩放素材。所有图在写入前都会
校验最终可见框数等于随机目标值。

```powershell
.\.venv\Scripts\python.exe -m classic_dataset.generate --config configs\dense_extension.yml --overwrite
.\.venv\Scripts\python.exe -m classic_dataset.validate --dataset 'F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\dense_extension'
```

增强集复用原 Pilot 的地图拆分，可用以下配置直接联合训练：

```text
F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\dense_extension\data_combined.yaml
```

当前增强集共 2,000 张、20,636 个框，1–20 框各档均有样本，密集场景 714 张；完整校验为零错误、
零警告。联合配置使用 Pilot 与增强集共 3,200 张 train 图片，val/test 固定使用原 Pilot。

## 金银岛 60K 全场景集

`comprehensive_60k` 是独立训练集，不与 Pilot 或 dense_extension 联合。它固定生成
48,000/6,000/6,000 张 train/val/test，覆盖 0–20 框配额、自然稀疏、多平台、密集重叠、
边缘裁切、前景遮挡、尺寸极值、低频动作，以及玩家/宠物/技能/伤害数字/掉落物/死亡动画等
无标签干扰物。177 张正常有怪地图按 141/18/18 拆分，单地图独有怪物只进入 train。

```powershell
# 120 张端到端烟雾集
.\.venv\Scripts\python.exe -m classic_dataset.comprehensive `
  --config configs\comprehensive_60k.yml --limit 120 `
  --name comprehensive_smoke_120 --overwrite

.\.venv\Scripts\python.exe -m classic_dataset.validate_comprehensive `
  --config configs\comprehensive_60k.yml `
  --dataset 'F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\comprehensive_smoke_120'

# 正式 60K；中断后重复执行同一命令即可续传
.\.venv\Scripts\python.exe -m classic_dataset.comprehensive `
  --config configs\comprehensive_60k.yml

.\.venv\Scripts\python.exe -m classic_dataset.validate_comprehensive `
  --config configs\comprehensive_60k.yml `
  --dataset 'F:\MapleStoryAssets\datasets\mxdclassic_monster_1280x224\comprehensive_60k'
```

输出包含 YOLO、COCO、`scenario_manifest.jsonl`、覆盖报告、natural/dense/edge/occlusion/
clutter/negative 挑战切片和分层可视化检查图。每张图先写临时文件再原子替换；Manifest 每条
强制落盘；F 盘可用空间低于 5 GB 时会安全停止。JPEG 使用质量 88 且关闭色度子采样，以便在
保持原生 `1280×224` 像素信息的同时控制 60K 数据体积。

## 48K 误检抑制与掉落物覆盖集

`hard_negative_48k_2class.yml` 在原 60K 两分类数据之外独立生成 48,000 张扩展图，类别仍只有
`monster` 和 `player`。宠物、掉落物、金币、死亡怪物、技能和伤害数字只作为未标注干扰物。
生成器会扫描客户端物品与装备图标、按 RGBA 外观去重，并用最低使用次数优先策略覆盖长尾物品；
同时控制每图掉落物数量、掉落堆布局、怪物数量和分散/聚集/重叠分布。

```powershell
# 1,000 张验收集
.\.venv\Scripts\python.exe -m classic_dataset.comprehensive `
  --config configs\hard_negative_48k_2class.yml --limit 1000 `
  --name hard_negative_pilot_1000 --overwrite

.\.venv\Scripts\python.exe -m classic_dataset.validate_comprehensive `
  --config configs\hard_negative_48k_2class.yml `
  --dataset 'N:\Program Files\map\hard_negative_pilot_1000'

# 验收后生成正式 48K；重复执行可从 Manifest 续传
.\.venv\Scripts\python.exe -m classic_dataset.comprehensive `
  --config configs\hard_negative_48k_2class.yml
```

正式输出位于 `N:\Program Files\map\hard_negative_48k_item_coverage`。其中
`data.yaml` 只读取新 48K，`data_combined_60k_plus_hard_negative.yaml` 则联合读取只读保留的原
60K 与新 48K，可直接用于下一轮两分类微调。`item_visual_index.json` 保存物品 ID 到独立视觉外观
的完整映射；`scenario_manifest.jsonl` 记录所有未标注干扰物的位置、可见比例、遮挡比例及其与
monster/player 的最大 IoU。

## 100 张分散场景验收样本

仅生成 100 张独立预览，供人工验收，不会启动正式 4 万张任务，也不划分 train/val/test。
配置为 `configs/scattered_preview_100.yml`，默认输出 `N:\Program Files\map\scattered_preview_100`。
输出目录非空时拒绝覆盖；复现时请先在配置中指定新的输出名称。

```powershell
.\.venv\Scripts\python.exe -m classic_dataset.scattered_preview --config configs\scattered_preview_100.yml
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_preview_integrity.py -v
```

- 分辨率 1280×224，类别 `0: monster`、`1: player`，宠物不标注。
- 场景配额依次为：人物+宠物+怪物 70 张、人物+怪物 15 张、人物+宠物 8 张、仅怪物 5 张、背景 2 张。
- 有怪物的图片含 1–5 只，怪物框间至少保留 48 像素的水平或垂直间隔；其他合成角色框间至少 24 像素。同图宠物 ID 不重复。
- 人物包含连接成功的头、身体、手臂、脸、头发、上衣、裤子和鞋，所有目标完整入镜。`catalogs` 保存可核查的人物与宠物外观及来源。
- `images` 为无框 PNG，`labels` 为 YOLO 标签，`visual_checks` 为画框图，`overviews` 为 10 页总览。打开 `index.html` 可浏览全部样本。
- `validation_report.json` 检查数量、尺寸、类别、标签与场景清单一致性、重复图片、入镜范围和间距。图像为客户端素材离线合成，最终视觉标准仍以人工验收为准。

本次修正了 WZSS v4/v5 的路径 GUID 对应、多页图集页码映射及裁切后的锚点；地图背景按视差与重复偏移定位，翻转物件保留原世界锚点。旧数据集不会自动重绘或修改。

## 正式 40K 分散两分类数据集

验收样本通过后，正式任务使用 `configs/scattered_40k_2class.yml`，输出到
`M:\Program Files\mxd\数据集\scattered_monster_player_40k`。数据按地图隔离为
32,000 train、4,000 val、4,000 test，类别仍为 `0: monster`、`1: player`，宠物不标注。

```powershell
.\scripts\run_scattered_40k_2class.ps1
```

任务采用高质量 JPEG、原子写入图片和标签，并在每张图完成后同步追加 Manifest。任务中断后重复执行同一命令即可续传；配置、计划或地图拆分与已有目录不一致时会拒绝混写。磁盘剩余空间低于 8 GB 时会安全停止。
