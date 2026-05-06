use color_eyre::eyre::Result;
use std::fs::File;
use std::io::Read;

use crate::consts::*;
use crate::matmul::MatMul;

fn to_float16(value: usize, offset: usize) -> usize {
    let neg = value > FIXED_POINT_MASK / 2;
    let abs = if neg {
        (FIXED_POINT_MASK + 1 - value) & (FIXED_POINT_MASK / 2)
    } else {
        value
    };
    (0..FIXED_POINT_SIZE)
        .rev()
        .find_map(|i| {
            if (abs >> i) & 1 > 0 {
                let res = ((abs << (FIXED_POINT_SIZE - i)) >> 14) & ((1 << 10) - 1);
                Some(res + ((i + 9 - offset) << 10) + ((if neg { 1 } else { 0 }) << 15))
            } else {
                None
            }
        })
        .unwrap_or(0)
}

fn float_mult(a: usize, b: usize, shift: usize) -> usize {
    let a_neg = a >= (1 << 15);
    let b_neg = b >= (1 << 15);
    let a_abs = a & ((1 << 15) - 1);
    let b_abs = b & ((1 << 15) - 1);
    let offset = ((a >> 10) & 31) + ((b >> 10) & 31);
    let neg = a_neg ^ b_neg;
    let a = if a_abs > 0 {
        (a_abs & ((1 << 10) - 1)) + (1 << 10)
    } else {
        0
    } as u128;
    let b = if b_abs > 0 {
        (b_abs & ((1 << 10) - 1)) + (1 << 10)
    } else {
        0
    } as u128;
    let res = (((a * b) << offset) >> (56 + shift)) as usize & FIXED_POINT_MASK;
    if neg {
        (FIXED_POINT_MASK + 1 - res) & FIXED_POINT_MASK
    } else {
        res
    }
}
pub struct Attention {
    matmul_key: [MatMul<EMBED_SIZE, HEAD_SIZE, false>; HEADS],
    matmul_value: [MatMul<EMBED_SIZE, HEAD_SIZE, false>; HEADS],
    matmul_query: [MatMul<EMBED_SIZE, HEAD_SIZE, false>; HEADS],
    matmul_proj: MatMul<EMBED_SIZE, EMBED_SIZE, false>,

    softmax_exps: Box<[usize; 1024]>,
}

impl Attention {
    pub fn new(block_num: usize) -> Result<Self> {
        // (HEADS, HEAD_SIZE, EMBED_SIZE)の3次元配列を作る
        let mut key: Box<[[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS]> =
            vec![[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS]
                .into_boxed_slice()
                .try_into()
                .unwrap();
        let mut value: Box<[[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS]> =
            vec![[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS]
                .into_boxed_slice()
                .try_into()
                .unwrap();
        let mut query: Box<[[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS]> =
            vec![[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS]
                .into_boxed_slice()
                .try_into()
                .unwrap();
        let mut proj: Box<[[u8; EMBED_SIZE]; EMBED_SIZE]> = vec![[0; EMBED_SIZE]; EMBED_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        for i in 0..24 {
            let cur_weights = {
                let mut cur_weights = vec![0u8; 9600];
                let path = format!(
                    "weights/weight_files/attention/att_{}.bin",
                    1 + 24 * block_num + i
                );
                let mut file = File::open(path)?;
                file.read_exact(&mut cur_weights)?;
                cur_weights
            };
            for j in 0..HEADS {
                let w0 = cur_weights[EMBED_SIZE * (3 * j)..EMBED_SIZE * (3 * j + 1)].try_into()?;
                let w1 =
                    cur_weights[EMBED_SIZE * (3 * j + 1)..EMBED_SIZE * (3 * j + 2)].try_into()?;
                let w2 =
                    cur_weights[EMBED_SIZE * (3 * j + 2)..EMBED_SIZE * (3 * j + 3)].try_into()?;
                let w20 =
                    cur_weights[EMBED_SIZE * (3 * j + 20)..EMBED_SIZE * (3 * j + 21)].try_into()?;
                let w21 =
                    cur_weights[EMBED_SIZE * (3 * j + 21)..EMBED_SIZE * (3 * j + 22)].try_into()?;
                let w22 =
                    cur_weights[EMBED_SIZE * (3 * j + 22)..EMBED_SIZE * (3 * j + 23)].try_into()?;
                if i % 2 == 0 {
                    key[j][2 * i] = w0;
                    value[j][2 * i] = w1;
                    query[j][2 * i] = w2;
                    key[j][2 * i + 1] = w20;
                    value[j][2 * i + 1] = w21;
                    query[j][2 * i + 1] = w22;
                } else {
                    key[j][2 * i + 1] = w0;
                    value[j][2 * i + 1] = w1;
                    query[j][2 * i + 1] = w2;
                    key[j][2 * i] = w20;
                    value[j][2 * i] = w21;
                    query[j][2 * i] = w22;
                }
            }
            for j in 0..HEADS {
                let w15 = cur_weights[EMBED_SIZE * (j + 15)..EMBED_SIZE * (j + 16)].try_into()?;
                let w35 = cur_weights[EMBED_SIZE * (j + 35)..EMBED_SIZE * (j + 36)].try_into()?;
                if i % 2 == 0 {
                    proj[HEAD_SIZE * j + 2 * i] = w15;
                    proj[HEAD_SIZE * j + 2 * i + 1] = w35;
                } else {
                    proj[HEAD_SIZE * j + 2 * i + 1] = w15;
                    proj[HEAD_SIZE * j + 2 * i] = w35;
                }
            }
        }
        let matmul_key = key
            .iter()
            .map(|k| MatMul::<EMBED_SIZE, HEAD_SIZE, false>::new(Box::new(*k)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let matmul_value = value
            .iter()
            .map(|v| MatMul::<EMBED_SIZE, HEAD_SIZE, false>::new(Box::new(*v)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let matmul_query = query
            .iter()
            .map(|q| MatMul::<EMBED_SIZE, HEAD_SIZE, false>::new(Box::new(*q)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let matmul_proj = MatMul::<EMBED_SIZE, EMBED_SIZE, false>::new(proj);

        let mut softmax_exps: Box<[usize; 1024]> =
            vec![0; 1024].into_boxed_slice().try_into().unwrap();
        let path = "weights/weight_files/softmax.bin";
        let mut file = File::open(path)?;
        for i in 0..1024 {
            let mut buf = [0u8; 3];
            file.read_exact(&mut buf)?;

            softmax_exps[i] = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]) as usize;
        }

        Ok(Self {
            matmul_key,
            matmul_value,
            matmul_query,
            matmul_proj,
            softmax_exps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_new() -> Result<()> {
        let attention = Attention::new(1)?;
        Ok(())
    }

    #[test]
    fn test_to_float16() {
        assert_eq!(to_float16(10319979, 8), 56872);
        assert_eq!(to_float16(6285204, 4), 28158);
        assert_eq!(to_float16(11220205, 10), 54604);
    }

    #[test]
    fn test_float_mult() {
        assert_eq!(float_mult(2131, 10507, 1), 0);
        assert_eq!(float_mult(45985, 62876, 5), 2);
        assert_eq!(float_mult(26021, 62193, 1), 16616731);
        assert_eq!(float_mult(0, 62193, 1), 0);
    }
}
