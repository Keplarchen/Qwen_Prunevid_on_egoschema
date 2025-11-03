"""
Custom Qwen2.5-VL Model with PruneVid Stage 1 Integration

基于 TimeChat DTD 的实现思路，在 position_embeddings 生成之后执行 Stage 1
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

# Import the original Qwen2.5-VL model
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLPreTrainedModel,
    )
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        # 如果分离导入失败，我们会在使用时再处理
        Qwen2_5_VLModel = None
        Qwen2_5_VLPreTrainedModel = None
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration
        Qwen2_5_VLModel = None
        Qwen2_5_VLPreTrainedModel = None


class Qwen2_5_VLModelWithPruneVid(nn.Module):
    """
    Custom Qwen2_5_VLModel that applies PruneVid Stage 1 after position embeddings

    继承原始模型并重写 forward 方法，在 position_embeddings 生成之后插入 Stage 1
    """

    def __init__(self, original_model, stage1_merger=None, config=None):
        """
        Args:
            original_model: 原始的 Qwen2_5_VLModel 实例
            stage1_merger: Stage 1 的 SpatialTemporalTokenMerger 实例
            config: PruneVidConfig
        """
        super().__init__()
        # 使用 object.__setattr__ 来避免触发 __setattr__
        object.__setattr__(self, 'original_model', original_model)
        object.__setattr__(self, 'stage1_merger', stage1_merger)
        object.__setattr__(self, 'prunevid_config', config)
        object.__setattr__(self, 'stage1_stats', {})

        # config 是必需的
        object.__setattr__(self, 'config', original_model.config)

    def get_input_embeddings(self):
        return self.original_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.original_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Custom forward with Stage 1 integration

        在 position_embeddings 生成之后，应用 Stage 1 时空 token 合并
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查是否需要应用 Stage 1
        vision_start_token_id = self.config.vision_start_token_id
        should_apply_stage1 = (
            self.stage1_merger is not None
            and input_ids is not None
            and vision_start_token_id in input_ids[0]
            and video_grid_thw is not None
            and self.prunevid_config is not None
            and self.prunevid_config.enable_stage1
        )

        if not should_apply_stage1:
            # 不需要 Stage 1，直接调用原始 forward
            return self.original_model.forward(
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

        # ========================================================================
        # 以下代码复制自原始 Qwen2_5_VLModel.forward()，并在适当位置插入 Stage 1
        # ========================================================================

        # 安全地访问 gradient_checkpointing
        gradient_checkpointing = getattr(self.original_model, 'gradient_checkpointing', False)
        if gradient_checkpointing and self.original_model.training:
            if use_cache:
                # logger.warning_once(...)
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.original_model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self.original_model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.original_model.rotary_emb(hidden_states, position_ids)

        # ========================================================================
        # ⭐ Stage 1: 时空 Token 合并（在 position_embeddings 生成之后）
        # ========================================================================

        if should_apply_stage1:
            # 检测视觉 token 的位置
            vision_end_token_id = self.config.vision_end_token_id
            batch_size = input_ids.shape[0]

            # 假设 batch_size = 1（简化处理）
            if batch_size == 1:
                sample_input_ids = input_ids[0]
                vision_start_indices = (sample_input_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
                vision_end_indices = (sample_input_ids == vision_end_token_id).nonzero(as_tuple=True)[0]

                visual_token_indices = []
                for start_idx, end_idx in zip(vision_start_indices, vision_end_indices):
                    visual_token_indices.extend(range(start_idx + 1, end_idx))

                if len(visual_token_indices) > 0 and self.prunevid_config.verbose:
                    print(f"[Stage 1] Found {len(visual_token_indices)} visual tokens")

                # 应用 Stage 1 合并
                if len(visual_token_indices) > 0:
                    try:
                        (
                            hidden_states,
                            position_ids,
                            position_embeddings,
                            stats,
                        ) = self.stage1_merger(
                            hidden_states=hidden_states,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            video_grid_thw=video_grid_thw.tolist() if isinstance(video_grid_thw, torch.Tensor) else video_grid_thw,
                            visual_token_indices=visual_token_indices,
                            batch_idx=0,
                        )

                        self.stage1_stats = stats

                        # 更新 causal_mask（如果序列长度改变了）
                        if hidden_states.shape[1] != inputs_embeds.shape[1]:
                            # 重新计算 causal_mask
                            # TODO: 这里需要更精确的处理
                            if self.prunevid_config.verbose:
                                print(f"[Stage 1] Sequence length changed: {inputs_embeds.shape[1]} -> {hidden_states.shape[1]}")

                    except Exception as e:
                        if self.prunevid_config.verbose:
                            print(f"[Stage 1] Error during token merging: {e}")
                            import traceback
                            traceback.print_exc()
                        # 出错时继续使用原始的 hidden_states

        # ========================================================================
        # 继续原始的 forward 流程
        # ========================================================================

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.original_model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if gradient_checkpointing and self.original_model.training:
                layer_outputs = self.original_model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.original_model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, *args, **kwargs):
        return self.original_model._update_causal_mask(*args, **kwargs)

    def _gradient_checkpointing_func(self, *args, **kwargs):
        return self.original_model._gradient_checkpointing_func(*args, **kwargs)

    def __getattr__(self, name):
        """代理未定义的属性到原始模型"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 尝试从原始模型获取
            return getattr(self.original_model, name)


class Qwen2_5_VLForConditionalGenerationWithPruneVid(Qwen2_5_VLForConditionalGeneration):
    """
    Custom Qwen2_5_VLForConditionalGeneration with PruneVid support

    替换内部的 model 为 Qwen2_5_VLModelWithPruneVid
    """

    def __init__(self, config, stage1_merger=None, prunevid_config=None):
        super().__init__(config)

        # 用自定义模型替换原始的 model
        self.model = Qwen2_5_VLModelWithPruneVid(
            original_model=self.model,
            stage1_merger=stage1_merger,
            config=prunevid_config,
        )

        self.stage1_merger = stage1_merger
        self.prunevid_config = prunevid_config

    @classmethod
    def from_pretrained_with_prunevid(
        cls,
        model_path: str,
        stage1_merger=None,
        prunevid_config=None,
        **kwargs,
    ):
        """
        从预训练模型加载，并集成 PruneVid

        Args:
            model_path: 模型路径
            stage1_merger: Stage 1 合并器
            prunevid_config: PruneVid配置
            **kwargs: 传递给 from_pretrained 的其他参数

        Returns:
            model: Qwen2_5_VLForConditionalGenerationWithPruneVid实例
        """
        # 先加载原始模型
        original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **kwargs
        )

        # 替换 model 为自定义版本
        original_model.model = Qwen2_5_VLModelWithPruneVid(
            original_model=original_model.model,
            stage1_merger=stage1_merger,
            config=prunevid_config,
        )

        # 保存引用
        original_model.stage1_merger = stage1_merger
        original_model.prunevid_config = prunevid_config

        return original_model
