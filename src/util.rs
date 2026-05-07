use color_eyre::eyre::Result;
use std::fs::File;
use std::io::Read;

pub struct PRNG {
    state: usize,
}

impl PRNG {
    pub fn new(seed: usize) -> Self {
        Self { state: seed }
    }

    pub fn next(&mut self) -> usize {
        for _ in 0..256 {
            let next_bit = ((self.state >> 22) & 1) ^ ((self.state >> 17) & 1);
            self.state = (self.state << 1) & ((1 << 23) - 1);
            self.state |= next_bit;
        }
        self.state
    }
}

pub struct Prompt {
    tokens: Vec<String>,
}

impl Prompt {
    pub fn new() -> Result<Self> {
        // tokens.txt からトークンを読み込む
        let mut file = File::open("tokens.txt")?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let tokens = contents
            .lines()
            .map(|s| s.to_string().replace("_", " "))
            .collect();
        Ok(Self { tokens })
    }

    pub fn parse(&self, text: &str) -> Result<Vec<usize>> {
        // 最初にスペースを追加し全体を小文字にする
        let mut prompt = format!(" {text}").to_lowercase();
        let mut ans = vec![0];
        while prompt.len() > 0 {
            let mut maxlen = 0;
            let mut token_id = None;
            for (i, token) in self.tokens.iter().enumerate() {
                if prompt.starts_with(token) && token.len() > maxlen {
                    maxlen = token.len();
                    token_id = Some(i);
                }
            }
            if let Some(id) = token_id {
                ans.push(id);
                prompt = prompt[maxlen..].to_string();
            } else {
                return Err(color_eyre::eyre::eyre!(
                    "Failed to parse prompt: remaining text '{}'",
                    prompt
                ));
            }
        }
        Ok(ans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prng() {
        let mut rng = PRNG::new(100);
        let expected = [
            1368217, 5749788, 2092439, 6159409, 6029481, 7701366, 3985400, 5311974, 3759117,
            2269441,
        ];
        for &e in &expected {
            assert_eq!(rng.next(), e);
        }
    }

    #[test]
    fn test_prompt() {
        let prompt = Prompt::new().unwrap();
        let text = "Hello, how are you?";
        let token_ids = prompt.parse(text).unwrap();
        let expected = vec![0, 79, 5, 33, 54, 18, 25];
        assert_eq!(token_ids, expected);
    }
}
