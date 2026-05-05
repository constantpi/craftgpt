pub const LAYERS: usize = 6;
pub const HEADS: usize = 5;
pub const MLP_SCALE: usize = 4;
pub const EMBED_SIZE: usize = 240;
pub const HEAD_SIZE: usize = EMBED_SIZE / HEADS;
pub const VOCAB_SIZE: usize = 1920;
pub const OUTPUT_SIZE: usize = 8;

pub const FIXED_POINT_SIZE: usize = 24;
pub const FIXED_POINT_MASK: usize = (1 << FIXED_POINT_SIZE) - 1;
pub const MATMUL_FIXED_POINT: usize = 18;
pub const MATMUL_EXTRA_PRECISION: usize = 4;
pub const MATMUL_BIG_MASK: usize = (1 << (FIXED_POINT_SIZE + MATMUL_EXTRA_PRECISION)) - 1;

pub const LAYERNORM_CONST: usize = (1 << 32) / EMBED_SIZE;
pub const LAYERNORM_CONST_2: usize = 8663717;
pub const ATT_CONST: usize = 4331858;

pub const EPS: usize =
    (1e-5 * EMBED_SIZE as f64 * (1usize << (2 * MATMUL_FIXED_POINT)) as f64) as usize;
