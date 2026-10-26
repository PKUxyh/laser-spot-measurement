# 激光焦斑分析

读取光束相机导出的二维强度矩阵，估计焦斑椭圆、能量集中比例和归一化矢势 `a0`。输入可以是带表头的 CSV，或单帧、单通道的 `.tif` / `.tiff`。TIFF 像素值直接作为强度计数使用，不做 8 位转换或 0–1 归一化；多页和彩色 TIFF 会报错。程序不会读取 TIFF 元数据中的像素尺寸，需按实际成像倍率设置标定值。

## 使用

### Windows 便携版

[下载 FocusAnalysisPortable-win64.zip](https://github.com/PKUxyh/laser-spot-measurement/releases/latest/download/FocusAnalysisPortable-win64.zip)

完整解压 `FocusAnalysisPortable-win64.zip`，双击 `FocusAnalysis.exe` 即可运行，
无需安装 Python。请将 EXE 和 `_internal` 文件夹保存在一起。
`source` 包含完整源码、测试和构建脚本；`verification.json` 是打包后 EXE 的自检结果。

### 从源码重新打包

使用 Windows x64 和 Python 3.10，在源码目录执行以下命令。
构建环境独立于系统 Python，避免引入其他项目的可选依赖。

```powershell
python -m venv .build\venv
.\.build\venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\build_release.ps1
```

脚本先运行测试，再构建并运行 EXE 的单图/批量 GUI 自检，最后生成
`release/FocusAnalysisPortable-win64.zip`。发布包中的 `source/requirements-lock.txt`
记录本次所有依赖版本，可替代 `requirements-build.txt` 安装以复现环境。

### Python 脚本

```powershell
python -m pip install -r requirements.txt
python analysis.py "光斑.tiff"
python batch_analysis.py "存放光斑文件的目录"
python app_gui.py
python zernike_analysis.py "光斑.tiff"
```

`analysis.py` 处理一幅图，保存 `figure/laser_spot.png`，同时显示区域扫描曲线。`batch_analysis.py` 处理目录下所有 CSV/TIF/TIFF 文件，在 `figure/` 保存半径、`a0` 与能量比例、指向抖动的统计图。GUI 可选择文件并修改标定和激光参数。Zernike 脚本对强度分布做多项式展开。

GUI 中的四幅图各按 APS 单栏单图的 `3.375 × 2.35 in` 比例绘制，窗口缩小时保持该比例；图表区可纵向滚动，以维持可读的最小显示尺寸。每幅图下方都有独立的 Matplotlib 导航工具栏，可框选缩放、平移、前进/后退和恢复完整视图。前两幅光斑图使用二维高斯拟合中心作为 `(0, 0)`，x/y 坐标根据 GUI 中输入的单像素物理尺寸显示为 `μm`；修改像素尺寸后点击 **Recalculate** 可更新坐标。两图的强度色标均为从零开始的线性刻度。第一幅图以白色绘制 FWHM 拟合椭圆、以橙色绘制能量积分椭圆；第二幅图以白色和橙色分别绘制由实测强度直接得到的 `1/e²` 与 FWHM 等高线，其中阈值分别为背景强度加上局部峰值以上幅度的 `e⁻²` 和一半。第三幅图展示穿过拟合中心的水平、垂直截线；第四幅图展示穿过拟合中心并沿 FWHM 拟合椭圆长轴、短轴方向的截线。两幅截线图的横轴均为相对中心的位置（`μm`），纵轴均为按当前脉冲参数换算的 `a0`，满足 `a0 ∝ √强度计数`。GUI 不再计算或绘制半径扫描曲线；右侧仍显示当前积分半径下的单个能量比例。单击图表可选中它，右键选择 **Save image as…** 可在保存对话框中选目录、修改文件名，导出单幅 PNG、TIFF、SVG 或 PDF。GUI 在后台执行图像分析，并逐幅绘制图表；底部状态栏显示进度。可用 `Ctrl+O` 打开图像、`F5` 重新计算。

GUI 还支持批量模式。点击 **Open files** 可直接在文件选择器中多选 CSV/TIF/TIFF；点击 **Open folder** 后，GUI 会列出所选文件夹顶层的全部受支持图像并默认全选，用户可以通过 Ctrl/Shift 多选、**Select all** 或 **Clear** 自行决定本次处理的文件。文件按名称中的数字自然排序并依次编号为 shot 1、2、3……；四幅批量图分别显示 `a0-shot`、`w_x/w_y-shot`、高斯拟合中心相对平均中心的 x/y 指向抖动，以及所有成功分析图像的平均强度伪彩图。前两幅曲线图的纵轴根据当前数据范围自动缩放，并在上下两侧留出边距。右侧显示平均 `a0`、平均 `w_x/w_y` 及各自的样本标准差（1σ），高斯中心在 x/y 方向的位置抖动（1σ），以及径向 RMS pointing jitter `sqrt(mean(Δx²+Δy²))`，单位均为 `μm`。平均图要求输入图像尺寸相同；无法分析的文件会被跳过并计入失败数。

GUI 的椭圆以实测半高等高线拟合，中心固定在二维高斯拟合中心。结果栏的 X/Y FWHM 是该椭圆穿过中心的水平/垂直全宽，长短轴长度也是 FWHM **全宽**；角度是长轴相对 x 正方向的角度，范围为 −90° 到 90°；椭圆度为长轴/短轴，圆形为 1。采用高斯强度分布假设，将 FWHM 半轴乘以 `sqrt(2/ln 2)` 得到 1/e² 半轴，用于前两幅叠加图、积分区域、能量比例和 `a0` 计算；X/Y 1/e² 半径则是该椭圆沿 x/y 中心截线的半宽。拟合效果显示为 **椭圆重合度**：实测半高区域与拟合椭圆内部的交集面积除以并集面积，范围 0–100%，越高代表轮廓形状越接近椭圆。它不衡量高斯强度曲面的拟合残差。

## 焦斑处理流程

1. 读取 CSV 或 TIFF 为二维浮点数组；将 NaN、无穷大及负值设为零。
2. 用 5×5 全 1 卷积核求局部强度和，以局部和最大的像素定位主光斑附近。
3. 以原始图像峰值的 `0.135` 倍作为阈值，生成二值掩膜；只保留包含上述像素的连通区域。
4. `skimage.measure.regionprops` 在二值区域上求几何质心、方向和等效椭圆长短轴。返回的半径为长短轴长度的一半；代码还将长短轴投影到 x/y 方向，得到 `w_x`、`w_y`，并以长轴半径除以短轴半径作为椭圆率。这里的椭圆半径是阈值区域的几何估计。GUI 以几何质心为初值，在主光斑附近拟合带背景项和旋转角的二维高斯函数，然后对实测半高等高线拟合椭圆，并将中心固定在高斯拟合中心。命令行与批处理仍使用几何质心和 13.5% 阈值区域的椭圆。5×5 局部和最大值只用于选择主光斑连通区域。
5. 以识别出的椭圆为内部区域，以半轴扩大 `region_total` 倍的椭圆为积分区域。能量集中比例 `η = 内部像素计数之和 / 外部像素计数之和`。命令行入口仍可显示积分区域倍数扫描曲线；GUI 只计算当前 `region_total` 对应的比例。
6. `analysis.py` 和批处理使用高斯近似：`I = 2ηE / (τπabΔxΔy)`；其中 `a,b` 是像素半轴，`ΔxΔy` 是平方厘米计的单像素面积，`E` 是焦斑能量，`τ` 是脉冲时长。GUI 使用拟合中心附近的 3×3 像素计数均值与外部椭圆计数总和之比估算峰值强度；3×3 均值不用于寻找中心。两者再用 `a0 = sqrt(I λ_μm² / 1.37e18)` 换算。两个入口的 `a0` 估算方法不同，数值不能直接视作同一算法的结果。

`calculate_radius_threshold()` 和 `calculate_radius_fwhm()` 还提供穿过最亮像素的 x/y 截线阈值法与半高宽法，但主流程使用上面的连通域椭圆法。

## 参数与解释边界

`analysis.py` 默认像素尺寸为 `4.4/39.11 μm`、`region_total=8`、能量为 `3.3×0.5585 J`、脉宽为 `35 fs`、波长为 `800 nm`；GUI 自带另一组默认值，并以 `fs` 和 `nm` 显示脉宽、波长，计算前分别换算为秒和米。GUI 的能量输入为靶上脉冲能量，单位为 J。请按相机像元尺寸、光学放大倍率和实验条件设置参数。CSV 沿用 `pandas.read_csv` 的默认表头规则；无表头 CSV 的首行会被当作列名。

Zernike 模块对**强度图像**而非波前相位做最高 10 阶的逐项投影和重建，所得系数不能直接解释为波前像差系数。
