"""
PruneVid集成Qwen2.5-VL模型
===========================

这个模块通过wrapper方式将PruneVid的3个阶段集成到Qwen2.5-VL模型中。

设计思路：
1. 不直接修改transformers的Qwen2.5-VL源码
2. 通过wrapper类包装原始模型
3. 在关键位置插入PruneVid的3个stage
4. 保持与transformers API的兼容性

集成点：
- Stage 1: 在vision encoder之后，LLM处理之前
- Stage 2: 在LLM的第M层通过hook提取注意力
- Stage 3: 通过自定义cache在generate时生效

注意事项：
由于Qwen2.5-VL的架构特殊性（vision encoder集成在模型内部），
我们需要在合适的位置拦截和修改token流。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import PruneVidConfig
from stage1_temporal_spatial_merge import SpatialTemporalTokenMerger
from stage2_attention_selection import AttentionBasedTokenSelector
from stage3_kv_cache import PruneVidDynamicCache


class Qwen2VLForConditionalGenerationWithPruneVid(nn.Module):
    """
    集成PruneVid的Qwen2.5-VL模型

    这个类包装原始的Qwen2VLForConditionalGeneration，
    在forward和generate流程中集成PruneVid的3个stage。
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: PruneVidConfig,
    ):
        """
        初始化

        Args:
            base_model: 预训练的Qwen2.5-VL模型
            config: PruneVid配置
        """
        super().__init__()

        self.base_model = base_model
        self.config = config

        # 初始化3个stage
        self.stage1 = SpatialTemporalTokenMerger(config) if config.enable_stage1 else None
        self.stage2 = AttentionBasedTokenSelector(config) if config.enable_stage2 else None

        # Stage 3在generate时通过past_key_values参数传入

        # 统计信息
        self.stats = {}

        # 特殊token IDs（用于检测视觉token位置）
        # Qwen2.5-VL使用特殊token标记视觉内容的开始和结束
        self.vision_start_token_id = getattr(base_model.config, 'vision_start_token_id', None)
        self.vision_end_token_id = getattr(base_model.config, 'vision_end_token_id', None)

        if self.config.verbose:
            print(f"[PruneVid] 初始化完成")
            print(f"  Stage 1: {'启用' if config.enable_stage1 else '禁用'}")
            print(f"  Stage 2: {'启用' if config.enable_stage2 else '禁用'}")
            print(f"  Stage 3: {'启用' if config.enable_cache_compression else '禁用'}")

    def _detect_visual_tokens(self, input_ids: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        检测视觉token的位置

        Qwen2.5-VL使用特殊token来标记视觉内容：
        [text...] <vision_start> [visual tokens...] <vision_end> [text...]

        Args:
            input_ids: [batch, seq_len] 输入token IDs

        Returns:
            visual_start: 视觉token起始位置（vision_start之后）
            visual_end: 视觉token结束位置（vision_end之前）
            如果没有检测到视觉token，返回(None, None)
        """
        if self.vision_start_token_id is None or self.vision_end_token_id is None:
            # 如果没有配置特殊token，尝试常见的ID
            # 这是一个fallback，实际使用中应该从config获取
            if self.config.verbose:
                print("[PruneVid] 警告：未找到vision token IDs，将尝试启发式检测")
            return None, None

        batch_size = input_ids.shape[0]
        if batch_size != 1:
            # 目前只支持batch=1
            if self.config.verbose:
                print(f"[PruneVid] 警告：batch_size={batch_size} > 1，暂不支持")
            return None, None

        input_ids_single = input_ids[0]  # [seq_len]

        # 查找vision_start和vision_end
        start_positions = (input_ids_single == self.vision_start_token_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids_single == self.vision_end_token_id).nonzero(as_tuple=True)[0]

        if len(start_positions) == 0 or len(end_positions) == 0:
            # 没有找到视觉token
            return None, None

        # 取第一对
        visual_start = start_positions[0].item() + 1  # +1因为要跳过start token本身
        visual_end = end_positions[0].item()  # end token之前

        if visual_start >= visual_end:
            if self.config.verbose:
                print(f"[PruneVid] 警告：无效的视觉token范围 [{visual_start}, {visual_end})")
            return None, None

        if self.config.verbose:
            print(f"[PruneVid] 检测到视觉tokens: [{visual_start}, {visual_end}), 数量={visual_end - visual_start}")

        return visual_start, visual_end

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward方法 - 集成PruneVid

        流程：
        1. 调用基础模型的embedding层
        2. 检测视觉token位置
        3. Stage 1: 时空token合并
        4. Stage 2: 设置attention hook（在指定层提取注意力）
        5. 调用基础模型的主体（会触发Stage 2）
        6. 返回结果

        注意：这是一个简化实现。完整实现需要深入修改forward流程。
        """
        # 简化实现：直接调用基础模型
        # 完整实现需要拦截中间层，这里我们通过generate方法来实现主要功能

        # 对于training/evaluation，暂时不启用PruneVid（需要更复杂的集成）
        if labels is not None:
            if self.config.verbose:
                print("[PruneVid] 训练模式，禁用PruneVid")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

        # 对于inference，也先调用基础模型
        # Stage 2需要在特定层提取注意力，这需要hook
        if self.config.enable_stage2 and output_attentions is None:
            output_attentions = True

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> torch.LongTensor:
        """
        生成方法 - 完整集成PruneVid的3个stage

        这是PruneVid主要生效的地方。

        流程：
        1. Prefill阶段：
           a. 检测视觉token位置
           b. Stage 1: 合并视觉token（需要在embedding后拦截）
           c. Stage 2: 在第M层提取注意力并选择token
        2. Decode阶段：
           a. Stage 3: 使用压缩的KV cache

        注意：完整实现Stage 1需要修改模型内部的forward流程。
        当前版本通过processor预处理来近似实现。
        """
        # 检测视觉token
        visual_start, visual_end = self._detect_visual_tokens(input_ids)

        # 准备past_key_values（Stage 3）
        if self.config.enable_cache_compression:
            past_key_values = PruneVidDynamicCache(verbose=self.config.verbose)
        else:
            past_key_values = None

        # 设置Stage 2的hook
        if self.config.enable_stage2 and visual_start is not None:
            target_layer = self.base_model.model.language_model.layers[self.config.pruning_layer]
            self.stage2.setup_hook(target_layer, visual_start, visual_end)

            if self.config.verbose:
                print(f"[PruneVid] Stage 2 hook已设置在layer {self.config.pruning_layer}")

        # 调用基础模型的generate
        # 注意：为了让attention hook工作，需要output_attentions=True（在prefill阶段）
        if 'output_attentions' not in generate_kwargs and self.config.enable_stage2:
            generate_kwargs['output_attentions'] = True

        # 传递自定义cache
        if past_key_values is not None:
            generate_kwargs['past_key_values'] = past_key_values

        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
        )

        # 清理hook
        if self.config.enable_stage2:
            self.stage2.remove_hook()

        # 收集统计信息
        if self.config.collect_stats:
            stats = {}
            if self.stage1 is not None:
                stats['stage1'] = self.stage1.last_stats
            if self.stage2 is not None:
                stats['stage2'] = self.stage2.last_stats
            if isinstance(past_key_values, PruneVidDynamicCache):
                stats['stage3'] = past_key_values.get_compression_stats()
            self.stats = stats

        return outputs

    def get_stats(self) -> Dict:
        """获取最近一次推理的统计信息"""
        return self.stats

    @property
    def device(self):
        """获取模型设备"""
        return self.base_model.device

    def to(self, *args, **kwargs):
        """移动模型到指定设备"""
        self.base_model = self.base_model.to(*args, **kwargs)
        return self

    def eval(self):
        """设置为评估模式"""
        self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        """设置训练模式"""
        self.base_model.train(mode)
        return self


def load_prunevid_model(
    model_name_or_path: str,
    config: Optional[PruneVidConfig] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    加载集成PruneVid的Qwen2.5-VL模型（使用DTD-based实现）

    Args:
        model_name_or_path: 模型路径或HuggingFace ID
        config: PruneVid配置，如果为None则使用baseline（不剪枝）
        device: 设备
        torch_dtype: 数据类型

    Returns:
        model: 集成PruneVid的模型
        processor: 对应的processor
    """
    from config import get_baseline_config
    from modeling_qwen2_5_vl_prunevid_dtd import Qwen2_5_VLForConditionalGeneration
    from configuration_qwen2_5_vl import Qwen2_5_VLConfig

    if config is None:
        config = get_baseline_config()

    # 直接使用transformers加载，然后包装
    print(f"加载Qwen2.5-VL模型: {model_name_or_path}")

    # 先直接加载预训练模型到目标设备
    from transformers import AutoModelForVision2Seq
    pretrained_model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 使用FlashAttention2 (最优性能)
    )

    # 创建我们的PruneVid模型，使用相同的config
    model = Qwen2_5_VLForConditionalGeneration(pretrained_model.config, prunevid_config=config)

    # 🔧 关键修复：转换权重keys以匹配PruneVid的模型结构
    # transformers的原始结构: model.visual.*, model.language_model.*
    # PruneVid的结构: visual.*, model.*
    def convert_state_dict_keys(state_dict):
        """转换transformers的state_dict keys到PruneVid格式"""
        new_state_dict = {}
        for key, value in state_dict.items():
            # 转换visual部分: model.visual.* -> visual.*
            if key.startswith('model.visual.'):
                new_key = key.replace('model.visual.', 'visual.')
            # 转换language model部分: model.language_model.* -> model.*
            elif key.startswith('model.language_model.'):
                new_key = key.replace('model.language_model.', 'model.')
            # 其他保持不变
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    # 转换原始模型的state_dict
    print("[PruneVid] Converting state_dict keys to match PruneVid structure...")
    converted_state_dict = convert_state_dict_keys(pretrained_model.state_dict())

    # 复制权重（pretrained_model已经在目标设备上）
    # 使用strict=True确保权重完全匹配，避免静默的权重加载错误
    try:
        model.load_state_dict(converted_state_dict, strict=True)
        print("[PruneVid] ✅ Successfully loaded all weights with strict=True")
    except RuntimeError as e:
        print(f"[PruneVid WARNING] Failed to load weights with strict=True: {e}")
        print("[PruneVid] Falling back to strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        if missing_keys:
            print(f"[PruneVid WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[PruneVid WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    # 确保模型在正确的设备上
    if device != "cpu":
        model = model.to(device)

    model.eval()

    # 删除pretrained_model释放内存
    del pretrained_model
    import gc
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # 加载processor
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    print(f"PruneVid模型加载完成")
    print(f"  配置: {config.to_dict()}")

    return model, processor


# 导出
__all__ = [
    "Qwen2VLForConditionalGenerationWithPruneVid",
    "load_prunevid_model",
]
