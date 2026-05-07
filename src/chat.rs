use color_eyre::eyre::Result;

use crate::model::Model;
use crate::util::{PRNG, Prompt};

pub struct Chat {
    model: Model,
    prompt: Prompt,
    prng: PRNG,
}

impl Chat {
    pub fn new(seed: usize) -> Result<Self> {
        Ok(Self {
            model: Model::new()?,
            prompt: Prompt::new()?,
            prng: PRNG::new(seed),
        })
    }

    pub fn invoke(&mut self, text: &str) -> Result<String> {
        let token_ids = self.prompt.parse(text)?;
        let mut output = String::new();
        for token_id in token_ids {
            self.model.process(token_id)?;
        }
        let mut nxt = 1;
        loop {
            let act = self.model.process(nxt)?;
            let mut random_value = self.prng.next();
            let mut here = None;
            for act_i in act.iter().rev() {
                let p = act_i >> 11;
                let r = act_i % 2048;
                if p < (1 << 20) {
                    continue;
                }
                if p > random_value {
                    here = Some(r);
                    break;
                }
                random_value -= p;
            }
            nxt = if let Some(here) = here {
                here
            } else {
                act[0] % 2048
            };
            if nxt == 0 || nxt == 1 {
                // 終了トークンが出たらループを抜ける
                break;
            }
            let token = self.prompt.get_token(nxt).unwrap_or("<UNK>");
            output.push_str(token);
        }
        Ok(output)
    }
}
