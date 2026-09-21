"""Single source of truth for v3 operator visibility and descriptive metadata."""


OPERATOR_METADATA = {
    "adaptive_threshold": ("1.0.0", "按局部邻域阈值分割，适合不均匀照明。", True),
    "bilateral_denoise": ("1.0.0", "边缘保持去噪。", True),
    "component_statistics": ("1.0.0", "计算连通域面积、质心和形状统计。", True),
    "convex_hull": ("1.0.0", "修复碎裂或不规则候选区域。", True),
    "hysteresis_threshold": ("1.0.0", "保留强响应及其连接的弱响应。", True),
    "invert_intensity": ("1.0.0", "反转稳健归一化后的图像极性。", True),
    "local_contrast": ("1.0.0", "CLAHE 局部对比度增强。", True),
    "local_background_residual": ("1.0.0", "从局部背景中提取亮、暗或绝对残差。", True),
    "median_denoise": ("1.0.0", "中值去噪，保留边缘。", True),
    "morphological_residual": ("1.0.0", "白顶帽或黑顶帽残差增强。", True),
    "normalize": ("1.0.0", "按图像分位数归一化灰度。", True),
    "percentile_clip": ("1.0.0", "裁剪极端灰度值。", True),
    "gaussian_denoise": ("1.0.0", "高斯去噪。", True),
    "global_threshold": ("1.0.0", "按亮度或暗度生成初始 Mask。", True),
    "morphology": ("1.0.0", "开闭、膨胀或腐蚀清理 Mask。", True),
    "fill_holes": ("1.0.0", "填充目标内部封闭空洞。", True),
    "filter_components": ("1.0.0", "按面积和形状筛选连通域。", True),
    "extract_contours": ("1.0.0", "从 Mask 提取闭合轮廓。", True),
    "remove_border_components": ("1.0.0", "移除接触边界的候选区域。", True),
    "remove_small_objects": ("1.0.0", "移除小型孤立候选。", True),
    "statistical_threshold": ("1.0.0", "使用 Otsu、Yen、Li、Triangle 或均值阈值分割。", True),
    "unsharp_enhance": ("1.0.0", "增强边缘和局部纹理对比度。", True),
    "exclude_regions": ("1.0.0", "生成排除边界、比例尺或指定矩形的有效区域 Mask。", True),
    "period_estimation": ("1.0.0", "估计重复结构的方向和周期。", True),
    "build_periodic_background": ("1.0.0", "根据图像和周期分析生成周期背景模型。", True),
    "subtract_periodic_background": ("1.0.0", "从原图减去周期背景，得到异常残差。", True),
    "build_periodic_valid_mask": ("1.0.0", "根据周期边缘和可选 ROI 创建有效区域 Mask。", True),
    "residual_threshold": ("1.0.0", "用分位数、Otsu 或 MAD 阈值分割残差。", True),
    "threshold_residual": ("1.0.0", "在有效区域内对残差进行确定性阈值分割。", True),
    "apply_mask_constraint": ("1.0.0", "将有效区域 Mask 确定性应用到候选 Mask。", True),
    # Legacy only: v3 uses the explicitly typed multi-input replacements above.
    "periodic_background_model": ("1.0.0", "legacy periodic background model", False),
    "periodic_background_residual": ("1.0.0", "legacy periodic background residual", False),
    "apply_valid_mask": ("1.0.0", "legacy valid-mask application", False),
}


def apply_operator_catalog(registry):
    for name in registry.names():
        version, description, model_visible = OPERATOR_METADATA.get(
            name,
            (registry.definition(name).version, "User-approved generated CV operator.", False),
        )
        registry.configure(
            name,
            version=version,
            description=description,
            model_visible=model_visible,
        )
    return registry


def model_visible_operator_names():
    return frozenset(
        name for name, (_, _, model_visible) in OPERATOR_METADATA.items() if model_visible
    )
