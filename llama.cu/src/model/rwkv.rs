use super::GGufModel;
use crate::utils::meta;
use nn::{
    Linear, Normalization, NormType, OutputHead, RWKV, RWKVBlock, Tensor, TimeMix, ChannelMix,
};

impl GGufModel<'_> {
    /// 构造 RWKV 模型
    pub fn rwkv(&self) -> nn::RWKV<Tensor<&[u8], 2>> {
        let n_layer = meta![self => block_count];
        let hidden_size = meta![self => hidden_size];
        let vocab_size = meta![self => vocab_size];
        let epsilon = meta![self => layer_norm_epsilon; 1e-5];
        
        let dt_linear = self.tensors["emb.weight"].dt();
        
        let get = |name: &str| self.tensors[name].as_deref();
        
        let emb_weight = get("emb.weight");
        let ln_out_weight = get("ln_out.weight");
        let head_weight = get("head.weight");

        RWKV {
            embedding: Embedding {
                dt: emb_weight.dt(),
                d: hidden_size,
                wte: Table {
                    row: vocab_size,
                    weight: emb_weight,
                },
                wpe: None,
            },
            blks: (0..n_layer)
                .map(|iblk| {
                    RWKVBlock::new(
                        Normalization {
                            d: hidden_size,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: ln_out_weight.dt(),
                                scale: get(&format!("blocks.{iblk}.ln1.weight")),
                            },
                        },
                        TimeMix {
                            k: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.time_mix_k.weight")),
                                None,
                            ),
                            v: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.time_mix_v.weight")),
                                None,
                            ),
                            r: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.time_mix_r.weight")),
                                None,
                            ),
                        },
                        Normalization {
                            d: hidden_size,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: ln_out_weight.dt(),
                                scale: get(&format!("blocks.{iblk}.ln2.weight")),
                            },
                        },
                        ChannelMix {
                            k: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.channel_mix_k.weight")),
                                None,
                            ),
                            r: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.channel_mix_r.weight")),
                                None,
                            ),
                            v: Linear::new(
                                dt_linear,
                                [hidden_size, hidden_size],
                                get(&format!("blocks.{iblk}.channel_mix_v.weight")),
                                None,
                            ),
                        },
                    )
                })
                .collect(),
            output_head: Some(OutputHead {
                out_norm: Normalization {
                    d: hidden_size,
                    epsilon: epsilon as _,
                    items: NormType::RmsNorm {
                        dt: ln_out_weight.dt(),
                        scale: ln_out_weight,
                    },
                },
                lm_head: Linear::new(
                    head_weight.dt(),
                    [vocab_size, hidden_size],
                    head_weight,
                    None,
                ),
            }),
        }
    }

    /// 构造 RWKV 模型的状态缓存张量
    pub fn rwkv_state_cache<const N: usize>(&self) -> Tensor<usize, N> {
        let dt = self.tensors["emb.weight"].dt();
        let n_layer = meta![self => block_count];
        let hidden_size = meta![self => hidden_size];
        
        // RWKV 状态包含: [n_layer, 2, hidden_size] (time_mix 和 channel_mix 各一个状态)
        Tensor::from_dim_slice(dt, [n_layer, 2, hidden_size])
    }
}
