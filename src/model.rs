use color_eyre::eyre::Result;
use color_eyre::eyre::eyre;

use crate::block::Block;
use crate::consts::*;
use crate::embed::{Embedding, Unembedding};
use crate::layer_norm::LayerNorm;

pub struct Model {
    embedding: Embedding,
    blocks: [Block; 6],
    layer_norm: LayerNorm,
    unembedding: Unembedding,
    index: usize,
}

impl Model {
    pub fn new() -> Result<Self> {
        let embedding = Embedding::new()?;
        let blocks: [Block; 6] = (0..6)
            .map(|i| Block::new(i))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| eyre!("Failed to create blocks array"))?;
        let layer_norm = LayerNorm::new(13)?;
        let unembedding = Unembedding::new()?;
        Ok(Self {
            embedding,
            blocks,
            layer_norm,
            unembedding,
            index: 0,
        })
    }

    pub fn process(&mut self, token: usize) -> Result<Box<[usize; OUTPUT_SIZE]>> {
        let mut value = self.embedding.get_weights(token, self.index)?;
        for block in &mut self.blocks {
            value = block.forward(&value);
        }
        value = self.layer_norm.forward(&value);
        let output = self.unembedding.forward(&value);
        self.index += 1;
        Ok(output)
    }
}
