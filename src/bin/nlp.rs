use clap::Command;

fn main() -> anyhow::Result<()> {
    let cmd = build_cli();
    let matches = cmd.get_matches();
    match matches.subcommand() {
        Some(("gpt", _matches)) => todo!(),
        Some(("llama", _matches)) => todo!(),
        _ => unreachable!("oops"),
    }
}

fn build_cli() -> Command {
    let cmd = Command::new("nlp")
        .bin_name("nlp")
        .subcommand_required(true);
    let cmd = build_gpt(cmd);
    let cmd = build_llama(cmd);
    cmd
}

fn build_gpt(cmd: Command) -> Command {
    let gpt = Command::new("gpt");
    cmd.subcommand(gpt)
}

fn build_llama(cmd: clap::Command) -> Command {
    let llama = Command::new("llama");
    cmd.subcommand(llama)
}
