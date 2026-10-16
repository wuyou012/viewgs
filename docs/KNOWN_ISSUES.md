# 已确认问题清单

1. **环境分裂（已处理）**：原项目使用三个 Conda 环境；ViewCrafter 为 Python 3.9/PyTorch 1.13.1，InstantSplat 与 ViewGS 为 Python 3.10/PyTorch 2.1.2。现以 ViewCrafter 的兼容栈建立 `viewgs-unified`，并重新编译 ViewGS CUDA 扩展。
2. **几何来源不完全统一**：InstantSplat 用 MASt3R 生成最终 `points3D.ply`；ViewCrafter 内部另跑 DUSt3R 生成插帧轨迹。ViewGS 使用前者点云，却没有读取后者的 `camera_params_full.txt`。
3. **监督图与监督相机存在小偏差**：当前 ViewGS 根据 MASt3R 两端相机重新插值中间相机，而扩散图由 ViewCrafter/DUSt3R 轨迹生成。已测得旋转路径平均偏差约 0.14–0.57°、最大约 1.17°；归一化相机中心最大偏差可达 6.85%，焦距也有差异。
4. **重复计算**：同一对图片先由 MASt3R 重建，又在 ViewCrafter 内由 DUSt3R 重建，增加时间和显存占用。后续应考虑让 ViewCrafter直接消费 MASt3R 点云与相机。
5. **输入配对曾依赖人工目录**：容易让两阶段使用不同图片。新脚本从 InstantSplat 的 `images.txt` 自动提取同一对原图。
6. **扩散端点未用于监督**：25 帧中两端被去除，实际使用中间 23 帧；这是当前设计，不是脚本漏帧。
7. **输出与源码混杂**：ViewCrafter 有 62 项、ViewGS 有 114 项未提交变化，包含代码、备份、日志、模型与实验结果。新仓库只收编排与文档，避免误上传大文件和历史产物。
8. **硬编码路径/GPU**：旧脚本固定绝对路径与 GPU。新脚本通过 `.env` 和命令行参数统一配置。
9. **名称易混淆**：当前代码实际调用的是 **ViewCrafter**，不是 VideoCrafter。
10. **权重可复现性仍需加强**：权重文件很大，未上传 GitHub；SHA256 已记录，但后续仍需补充稳定下载地址。
11. **InstantSplat/PyTorch 版本兼容**：`init_geo.py` 无条件设置 PyTorch 2.x 的 `expandable_segments`，在统一环境的 PyTorch 1.13 上会报错。新流程用进程内兼容启动器移除此配置，不改动原仓库。
12. **ViewCrafter 两图输入会丢第一张**：当前 loader 固定把索引 0 当 LLFF 测试帧，直接传两张图时只剩一张训练图。新脚本在排序最前面放置一张临时重复图作为 holdout，使真实两端图都进入重建。
13. **512 配置的尺寸约束**：DUSt3R 输入日志显示 512×288，但 ViewCrafter 的 512 扩散配置必须使用 512×320 输出张量；错误设置为 512×288 会在 U-Net skip connection 处出现 10/9 尺寸不匹配。

