use clap::Command;

fn main() -> anyhow::Result<()> {
    let cmd = build_cli();
    let _matches = cmd.get_matches();
    Ok(())
}

fn build_cli() -> clap::Command {
    let cmd = Command::new("nlp")
        .bin_name("nlp")
        .subcommand_required(true);
    let cmd = build_gpt(cmd);
    let cmd = build_llama(cmd);
    cmd
}

fn build_gpt(cmd: Command) -> clap::Command {
    let gpt = Command::new("gpt");
    cmd.subcommand(gpt)
}

fn build_llama(cmd: clap::Command) -> clap::Command {
    let llama = Command::new("llama");
    cmd.subcommand(llama)
}
