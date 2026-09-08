# kira-ai-plugin-sticker-plus（增强表情包）

KiraAI 增强表情包插件，移植自 nori-core 的 `nori_plugin_emoji` 并按 KiraAI
插件规范重写。AI 通过 `send_emoji` 工具按情绪发图（VLM 候选择优），同时自动
收藏聊天中他人发送的表情包，形成越用越懂你的表情包图库。

> ⚠️ **使用前请先在 WebUI 插件页禁用内置「默认表情包」插件**，否则 AI 会同时
> 看到两条表情包通道（内置 `<sticker>` 标签清单 + 本插件的 send_emoji 工具），
> 行为不可预期。

## 功能

- **发送表情包**（`send_emoji` 工具）：AI 从 20 类固定情绪词表中给出情绪关键词，
  插件按情绪标签采样候选（6 张情绪匹配 + 随机补足），VLM 基于候选的描述文本
  选出最贴切的一张，以附件形式交给 AI 用 `<file type="image">` 发送
- **偷表情**（`@on.im_message` hook）：他人发送的市场表情 / 表情包自动入库
  （普通图片不入库），WebUI 管理页可一键开关
- **VLM 自动打标**：新表情入库后由 VLM 后台生成画面描述 + 情绪标签
  （胆怯、无语、调皮、开心、困惑、震惊、傲娇、害羞、温柔、委屈、期待、生气、
  无辜、撒娇、嫌弃、嘲讽、感谢、安慰、悲伤、欢迎）
- **防重复与淘汰**：最近 3 张 + 最高频 3 张表情本轮不发；库满按
  "已打标 → 使用最少 → 最久未用"（完全同分时最旧创建优先）淘汰，新入库表情
  不会被立即挤掉
- **入库质检**：非图片内容（损坏文件、视频贴纸等）在入库时即拒收；连续 3 次
  打标失败的条目自动禁用（可在管理页启用 + 重打标恢复）
- **WebUI 管理页**（侧边栏「增强表情包」）：表情包预览（点击放大）、描述与
  标签编辑、禁用/启用、删除、重新打标、多文件上传、目录重扫、偷取开关、
  库存统计

## 配置

配置文件位于 `data/config/plugins/kira-ai-plugin-sticker-plus.json`，
WebUI 插件配置页可视化编辑：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `steal_emoji` | `true` | 偷取聊天表情（管理页也可开关） |
| `capacity` | `500` | 图库容量上限，满额淘汰低使用量表情 |
| `candidate_count` | `9` | 每次 send_emoji 的 VLM 候选数 |
| `max_emoji_size_mb` | `5.0` | 偷取单图大小上限（MB） |
| `vlm_model` | 空 | 打标/选图模型，留空用系统默认 VLM（打标需视觉能力） |

## 数据与重置

全部状态存于插件数据目录 `data/plugin_data/kira-ai-plugin-sticker-plus/`：

- `emoji.db` — SQLite 库（表 `emoji_images`，插件自建自管，不碰宿主数据库）
- `emojis/` — 表情包图片（以 SHA256 命名）

删除该目录即完全重置。往 `emojis/` 直接放图片文件后点管理页「重新扫描」
即可批量导入。

## 依赖

无额外依赖（sqlalchemy / aiosqlite / Pillow / httpx 均为 KiraAI 宿主自带）。

## 测试

自带测试套件，需在 KiraAI 宿主检出内运行（`conftest.py` 会向上定位宿主根，
找不到则自动跳过收集）：

```bash
# 在 KiraAI 仓库根目录
python -m pytest data/plugins/kira-ai-plugin-sticker-plus/tests -q
```

## 已知降级

QQ 通道上表情包以普通图片消息发出（宿主 QQ 适配器把 Sticker/Image 统一编码为
base64 图片），对方看到的是图片而非"收藏表情"，这是平台协议限制，无法在
插件层修复。

## 来源与许可

移植自 nori-core 仓库 `plugins/nori_plugin_emoji`（MIT），核心算法
（情绪采样 / 防重复 / 淘汰策略 / 提示词）保持一致，宿主适配层
（工具注册 / hook / WebUI / 存储）按 KiraAI 插件规范重写。
