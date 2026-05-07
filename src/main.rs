mod attention;
mod block;
mod chat;
mod consts;
mod embed;
mod layer_norm;
mod matmul;
mod mlp;
mod model;
mod util;

use color_eyre::eyre::Result;

use crate::chat::Chat;

fn main() -> Result<()> {
    let mut chat = Chat::new(1)?;
    // let response = chat.invoke("Hello")?;
    // println!("{response}");
    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let response = chat.invoke(input.trim())?;
        println!("{response}");
    }
}
