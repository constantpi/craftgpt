use crate::consts::*;

#[derive(Debug)]
pub struct MatMul<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const RELU: bool> {
    weights: Box<[[(bool, u8, u8, u8); INPUT_SIZE]; OUTPUT_SIZE]>,
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
    pub fn new(weights: Box<[[u8; INPUT_SIZE]; OUTPUT_SIZE]>) -> Self {
        let mut decoded_weights: Box<[[(bool, u8, u8, u8); INPUT_SIZE]; OUTPUT_SIZE]> =
            vec![[(false, 0, 0, 0); INPUT_SIZE]; OUTPUT_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap();

        decoded_weights
            .iter_mut()
            .zip(weights.iter())
            .for_each(|(decoded_row, &row)| {
                decoded_row
                    .iter_mut()
                    .zip(row.iter())
                    .for_each(|(decoded_cell, &w)| {
                        *decoded_cell = decode_weight(false, w);
                    });
            });

        Self {
            weights: decoded_weights,
        }
    }

    pub fn forward(&self, input: &Box<[usize; INPUT_SIZE]>) -> Box<[usize; OUTPUT_SIZE]> {
        // 符号拡張関数
        fn sign_extend(x: usize) -> usize {
            let x = x & MATMUL_BIG_MASK;
            if x > MATMUL_BIG_MASK / 2 {
                x + (255 << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE))
            } else {
                x
            }
        }

        // 1. normed をヒープ上に確保し、ループで計算結果を書き込む
        let mut normed: Box<[usize; INPUT_SIZE]> =
            vec![0; INPUT_SIZE].into_boxed_slice().try_into().unwrap();

        normed
            .iter_mut()
            .zip(input.iter())
            .for_each(|(normed_i, &x)| {
                let abs = x & FIXED_POINT_MASK;
                *normed_i = if abs > FIXED_POINT_MASK / 2 {
                    abs + (((1 << MATMUL_EXTRA_PRECISION) - 1) << FIXED_POINT_SIZE)
                } else {
                    abs
                };
            });

        // 2. output をヒープ上に確保し、ループで計算結果を書き込む
        let mut output: Box<[usize; OUTPUT_SIZE]> =
            vec![0; OUTPUT_SIZE].into_boxed_slice().try_into().unwrap();

        output
            .iter_mut()
            .zip(self.weights.iter())
            .for_each(|(output_i, &row)| {
                let mut cur = 0;

                // self.weights[i] (1行分) と normed を zip して計算
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

                *output_i = if RELU && cur > (FIXED_POINT_MASK / 2) {
                    0
                } else {
                    cur
                };
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
            let matmul = MatMul::<3, 2, false>::new(Box::new(weights));
            let output = matmul.forward(&Box::new(input));
            assert_eq!(output, Box::new(ans));
        }

        let weights = [
            [
                25, 43, 194, 243, 2, 123, 191, 159, 214, 122, 19, 22, 103, 122, 145, 51, 15, 242,
                226, 97,
            ],
            [
                27, 73, 22, 26, 173, 197, 153, 108, 143, 12, 232, 113, 103, 183, 19, 60, 54, 192,
                188, 75,
            ],
            [
                149, 106, 16, 237, 171, 187, 25, 236, 74, 227, 153, 168, 245, 208, 137, 24, 110,
                199, 1, 149,
            ],
            [
                201, 222, 97, 172, 236, 171, 225, 234, 124, 215, 7, 249, 127, 210, 164, 192, 48,
                126, 90, 103,
            ],
            [
                166, 125, 106, 146, 246, 116, 242, 100, 78, 132, 231, 22, 107, 168, 75, 13, 196,
                200, 18, 14,
            ],
            [
                91, 24, 70, 120, 37, 246, 251, 217, 116, 23, 194, 18, 74, 213, 240, 205, 107, 57,
                54, 125,
            ],
            [
                55, 27, 200, 42, 208, 161, 11, 214, 72, 0, 237, 250, 237, 150, 71, 132, 80, 88,
                235, 70,
            ],
            [
                196, 28, 196, 68, 226, 97, 24, 235, 157, 70, 102, 213, 73, 195, 82, 217, 52, 60,
                235, 89,
            ],
            [
                221, 84, 147, 170, 150, 162, 174, 143, 29, 189, 114, 188, 248, 134, 24, 228, 249,
                206, 251, 93,
            ],
            [
                91, 62, 186, 61, 129, 73, 254, 55, 24, 104, 30, 232, 36, 178, 146, 6, 202, 38, 57,
                134,
            ],
        ];
        let input = [
            1237388, 11297805, 14601180, 12773817, 2145272, 15188795, 5978039, 3419513, 12410957,
            2715737, 4511459, 12908708, 9169872, 8396674, 13577819, 10958113, 6583058, 10023077,
            13014018, 11954757,
        ];

        let ans = [
            15364410, 15525973, 2167098, 9009896, 11416486, 12097014, 1266462, 268242, 2071641,
            14265011,
        ];
        let matmul = MatMul::<20, 10, false>::new(Box::new(weights));
        let output = matmul.forward(&Box::new(input));
        assert_eq!(output, Box::new(ans));
    }
}
