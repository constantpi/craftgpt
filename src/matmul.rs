use crate::consts::*;

pub struct MatMul<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const RELU: bool> {
    weights: [[(bool, u8, u8, u8); INPUT_SIZE]; OUTPUT_SIZE],
}

fn decode_weight(sign: bool, w: u8) -> (bool, u8, u8, u8) {
    match w {
        0..64 => (sign, 8, w / 8, w % 8),
        64..96 => (sign, 7, 4 + (w - 64) / 8, w % 8),
        96..112 => (sign, 5, 2 + (w - 96) / 8, w % 8),
        112..120 => (sign, 3, 1 + (w - 112) / 8, w % 8),
        120..128 => (sign, 2, 1 + (w - 120) / 8, w % 8),
        _ => decode_weight(true, w - 128),
    }
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const RELU: bool>
    MatMul<INPUT_SIZE, OUTPUT_SIZE, RELU>
{
    pub fn new(weights: [[u8; INPUT_SIZE]; OUTPUT_SIZE]) -> Self {
        Self {
            weights: weights.map(|row| row.map(|w| decode_weight(false, w))),
        }
    }

    pub fn forward(&self, input: &[usize; INPUT_SIZE]) -> [usize; OUTPUT_SIZE] {
        // 符号拡張関数
        fn sign_extend(x: usize) -> usize {
            if x > MATMUL_BIG_MASK / 2 {
                x + 255 << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE)
            } else {
                x
            }
        }

        let normed = input.map(|x| {
            let abs = x & FIXED_POINT_MASK;
            if abs > FIXED_POINT_MASK / 2 {
                x + (((1 << MATMUL_EXTRA_PRECISION) - 1) << FIXED_POINT_SIZE)
            } else {
                x
            }
        });

        let output = self.weights.map(|row| {
            let mut cur = 0;
            row.iter()
                .zip(normed.iter())
                .for_each(|(&(is_minus, w1, w2, w3), &x)| {
                    let w1 = w1 as usize;
                    let w2 = w2 as usize;
                    let w3 = w3 as usize;
                    let big = sign_extend(x * w2);
                    let small = sign_extend(x * w3);
                    let abs_cont = ((big >> w1) + (small >> (w1 + 3))) & FIXED_POINT_MASK;
                    let cont = if is_minus {
                        (FIXED_POINT_MASK + 1 - abs_cont) & FIXED_POINT_MASK
                    } else {
                        abs_cont
                    };
                    cur += cont;
                    cur &= FIXED_POINT_MASK;
                });
            if RELU && cur > (FIXED_POINT_MASK / 2) {
                0
            } else {
                cur
            }
        });

        output
    }
}

// test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let data = [
            (
                [[234, 174, 2], [193, 247, 190]],
                [69, 64, 207],
                [16777209, 16777194],
            ),
            ([[95, 184, 152], [166, 34, 8]], [254, 246, 19], [8, 0]),
            ([[125, 133, 77], [87, 199, 167]], [83, 2, 81], [35, 2]),
        ];
        for (weights, input, ans) in data {
            let matmul = MatMul::<3, 2, false>::new(weights);
            let output = matmul.forward(&input);
            assert_eq!(output, ans);
        }
    }
}
