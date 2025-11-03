"""
Qwen2.5-VL适配器

处理Qwen2.5-VL模型特定的集成：
- 检测视觉token位置（使用image_grid_thw）
- 集成三个阶段的token pruning
- 处理Grouped Query Attention (GQA)
- 管理forward hooks
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

# 导入Qwen2.5-VL模型（注意是Qwen2_5，不是Qwen2）
try:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration

try:
    from .config import PruneVidConfig
    from .stage1_merging import SpatialTemporalMerger, create_spatial_temporal_merger
    from .stage1_token_merge import SpatialTemporalTokenMerger, create_spatial_temporal_token_merger
    from .stage2_selection import AttentionBasedTokenSelector, register_attention_pruning_hook
    from .stage3_cache import PruneVidDynamicCache, create_prunevid_cache
except ImportError:
    from config import PruneVidConfig
    from stage1_merging import SpatialTemporalMerger, create_spatial_temporal_merger
    from stage1_token_merge import SpatialTemporalTokenMerger, create_spatial_temporal_token_merger
    from stage2_selection import AttentionBasedTokenSelector, register_attention_pruning_hook
    from stage3_cache import PruneVidDynamicCache, create_prunevid_cache


class Qwen25VLAdapter:
    """
    Qwen2.5-VL模型的PruneVid适配器

    负责：
    1. 检测视觉token在输入序列中的位置
    2. 协调三个阶段的token pruning
    3. 管理forward hooks和自定义cache
    """

    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        config: PruneVidConfig,
    ):
        """
        Args:
            model: Qwen2.5-VL模型实例
            config: PruneVid配置
        """
        self.model = model
        self.config = config
        config.validate()

        # 三个阶段的组件
        self.stage1_merger: Optional[SpatialTemporalMerger] = None
        self.stage2_selector: Optional[AttentionBasedTokenSelector] = None
        self.stage3_cache: Optional[PruneVidDynamicCache] = None

        # Hooks
        self.hooks = []

        # 统计信息
        self.last_stats = {}

        # 初始化各阶段
        self._initialize_stages()

    def _initialize_stages(self):
        """初始化各阶段的组件"""
        # Stage 1: 时空token合并（在位置编码之后执行）
        if self.config.enable_stage1:
            self.stage1_merger = create_spatial_temporal_token_merger(
                tau=self.config.tau,
                cluster_ratio=self.config.cluster_ratio,
                temporal_segment_ratio=self.config.temporal_segment_ratio,
                dpc_knn_k=self.config.dpc_knn_k,
                verbose=self.config.verbose,
            )

        # Stage 2: 基于注意力的token选择
        if self.config.enable_pruning:
            self.stage2_selector = AttentionBasedTokenSelector(
                pruning_layer=self.config.pruning_layer,
                keep_ratio=self.config.keep_ratio,
                aggregation=self.config.attention_aggregation,
                verbose=self.config.verbose,
            )

        # Stage 3: KV缓存压缩（跟随Stage 2）
        if self.config.enable_pruning:
            self.stage3_cache = create_prunevid_cache()

    def detect_visual_token_positions(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[List[List[int]]] = None,
    ) -> Tuple[int, int, int]:
        """
        检测视觉token在输入序列中的位置

        Qwen2.5-VL在处理时，视觉token会被插入到input_ids中的特殊位置。
        我们需要找到这些位置。

        Args:
            input_ids: [batch_size, seq_len] 输入token ids
            image_grid_thw: [[t, h, w], ...] 每个图像/视频的网格大小

        Returns:
            visual_token_start: 视觉token起始位置
            visual_token_end: 视觉token结束位置
            num_visual_tokens: 视觉token总数
        """
        if image_grid_thw is None or len(image_grid_thw) == 0:
            # 没有视觉输入
            return 0, 0, 0

        # 计算视觉token数量
        # 在Qwen2.5-VL中，视觉token数量 = sum(t * (h//2) * (w//2))
        # 因为PatchMerger会进行2x2的pooling
        num_visual_tokens = sum(
            t * (h // 2) * (w // 2) for t, h, w in image_grid_thw
        )

        # 在Qwen2.5-VL中，视觉token通常在序列的开始位置
        # 但具体位置取决于模型的实现
        # 这里我们假设视觉token在最前面

        # 一个更鲁棒的方法是查找vision_start_token_id和vision_end_token_id
        # 但为了简化，我们假设视觉token从位置0开始
        visual_token_start = 0
        visual_token_end = num_visual_tokens

        return visual_token_start, visual_token_end, num_visual_tokens

    def apply_stage1_to_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        image_grid_thw: List[List[int]],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        对视觉token应用Stage 1的时空合并

        Args:
            visual_tokens: [num_visual_tokens, hidden_dim] 视觉token
            image_grid_thw: [[t, h, w], ...] 每个视频的网格大小

        Returns:
            merged_tokens: [num_merged_tokens, hidden_dim] 合并后的token
            stats: 统计信息
        """
        if not self.config.enable_stage1 or self.stage1_merger is None:
            # Stage 1未启用，直接返回
            return visual_tokens, {"enabled": False}

        # 重塑为 [T, N, C] 格式
        # 假设只有一个视频
        if len(image_grid_thw) != 1:
            # 暂不支持多视频
            if self.config.verbose:
                print(f"[Stage 1] Warning: Multiple videos not supported, skipping Stage 1")
            return visual_tokens, {"enabled": False, "reason": "multiple_videos"}

        t, h, w = image_grid_thw[0]
        h_pooled = h // 2  # PatchMerger的2x2 pooling
        w_pooled = w // 2
        n_tokens_per_frame = h_pooled * w_pooled
        hidden_dim = visual_tokens.shape[-1]

        # 重塑为 [T, N, C]
        frame_tokens = visual_tokens.reshape(t, n_tokens_per_frame, hidden_dim)

        # 应用时空合并
        merged_tokens, stats = self.stage1_merger(frame_tokens)

        stats["enabled"] = True
        return merged_tokens, stats

    def register_hooks(
        self,
        visual_token_start: int,
        visual_token_end: int,
        video_grid_thw: Optional[List[List[int]]] = None,
        inputs_dict: Optional[Dict] = None,
    ):
        """
        注册forward hooks

        Args:
            visual_token_start: 视觉token起始位置
            visual_token_end: 视觉token结束位置
            video_grid_thw: 视频网格大小（用于Stage 1）
            inputs_dict: 输入字典，用于更新参数
        """
        # 清除已有的hooks
        self.remove_hooks()

        # 保存参数
        self._visual_token_start = visual_token_start
        self._visual_token_end = visual_token_end
        self._video_grid_thw = video_grid_thw
        self._inputs_dict = inputs_dict

        # Monkey patch Qwen2_5_VLModel.forward() 方法
        if self.config.enable_stage1 and self.stage1_merger is not None:
            self._patch_model_forward()

        # 注册Stage 2的pruning hook
        if self.config.enable_pruning and self.stage2_selector is not None:
            hook_handle = register_attention_pruning_hook(
                model=self.model,
                selector=self.stage2_selector,
                cache=self.stage3_cache,
                visual_token_start=visual_token_start,
                visual_token_end=visual_token_end,
                adapter=self,  # 传递adapter以便获取Stage 1更新后的位置
            )
            self.hooks.append(hook_handle)

    def _patch_model_forward(self):
        """
        Monkey patch Qwen2_5_VLModel.forward() 方法，
        在 position_embeddings 生成之后执行 Stage 1
        """
        original_forward = self.model.model.forward

        def patched_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            **kwargs,
        ):
            """
            Patched forward method with Stage 1 integration
            """
            # 调用原始 forward 直到生成 position_embeddings 之前
            # 由于需要访问中间变量，我们需要复制部分逻辑

            # 检查是否需要应用 Stage 1
            vision_start_token_id = self.model.model.config.vision_start_token_id
            should_apply_stage1 = (
                self.config.enable_stage1
                and self.stage1_merger is not None
                and input_ids is not None
                and vision_start_token_id in input_ids[0]
                and video_grid_thw is not None
            )

            if not should_apply_stage1:
                # 不需要 Stage 1，直接调用原始 forward
                return original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    **kwargs,
                )

            # 需要应用 Stage 1
            # 这里我们无法直接访问 forward 内部的局部变量
            # 所以我们使用一个 hook 的方式
            # 在 rotary_emb 之后插入 Stage 1

            # TODO: 这个实现比较复杂，需要仔细处理
            # 暂时先调用原始 forward，后续优化
            if self.config.verbose:
                print("[Stage 1] Warning: Monkey patch forward not fully implemented yet")

            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs,
            )

        self.model.model.forward = patched_forward
        self._original_model_forward = original_forward

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # 恢复被monkey patch的方法
        if hasattr(self, '_original_get_video_features'):
            self.model.model.get_video_features = self._original_get_video_features
            delattr(self, '_original_get_video_features')

        if hasattr(self, '_original_model_forward'):
            self.model.model.forward = self._original_model_forward
            delattr(self, '_original_model_forward')

        # 清理Stage 1的压缩信息和缓存
        if hasattr(self, 'compressed_visual_token_end'):
            delattr(self, 'compressed_visual_token_end')
        if hasattr(self, 'stage1_stats'):
            delattr(self, 'stage1_stats')
        if hasattr(self, '_original_video_grid_thw'):
            delattr(self, '_original_video_grid_thw')
        if hasattr(self, '_inputs_dict'):
            delattr(self, '_inputs_dict')
        if hasattr(self, '_cached_features'):
            delattr(self, '_cached_features')
        if hasattr(self, '_cache_valid'):
            delattr(self, '_cache_valid')

    def create_cache(self) -> PruneVidDynamicCache:
        """
        创建用于生成的自定义cache

        Returns:
            cache: PruneVidDynamicCache实例
        """
        if self.config.enable_pruning and self.stage3_cache is not None:
            # 重置cache
            self.stage3_cache.reset()
            return self.stage3_cache
        else:
            # 如果Stage 2/3未启用，返回None（使用默认cache）
            return None

    def get_statistics(self) -> Dict:
        """
        获取各阶段的统计信息

        Returns:
            stats: 统计信息字典
        """
        stats = {
            "config": {
                "stage1_enabled": self.config.enable_stage1,
                "stage2_enabled": self.config.enable_pruning,
                "stage3_enabled": self.config.enable_pruning,
            }
        }

        # Stage 1统计
        if self.stage1_merger is not None:
            # 优先使用hook中保存的统计信息
            if hasattr(self, 'stage1_stats'):
                stats["stage1"] = self.stage1_stats
            else:
                stats["stage1"] = self.stage1_merger.stats if hasattr(self.stage1_merger, 'stats') else {"enabled": False}

        # Stage 2统计
        if self.stage2_selector is not None:
            try:
                from .stage2_selection import get_pruning_statistics
            except ImportError:
                from stage2_selection import get_pruning_statistics
            stage2_stats = get_pruning_statistics(self.stage2_selector)
            if stage2_stats is not None:
                stats["stage2"] = stage2_stats

        # Stage 3统计
        if self.stage3_cache is not None:
            stats["stage3"] = self.stage3_cache.get_statistics()

        return stats

    def __del__(self):
        """清理资源"""
        self.remove_hooks()


def create_qwen_adapter(
    model: Qwen2_5_VLForConditionalGeneration,
    config: PruneVidConfig,
) -> Qwen25VLAdapter:
    """
    创建Qwen2.5-VL适配器

    Args:
        model: Qwen2.5-VL模型
        config: PruneVid配置

    Returns:
        adapter: Qwen25VLAdapter实例
    """
    return Qwen25VLAdapter(model, config)
