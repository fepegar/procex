import typer
from procex.io import read_image
from procex.io import write_image
from procex.functional import resize as resize_image
from procex.functional import resize as stretch_image


app = typer.Typer()

@app.command()
def resize(
    input_path,
    output_path,
    size,
    quality,
):
    typer.echo("TODO")


@app.command()
def stretch(
    input_path,
    output_path,
    num_bits: int | None = None,
):
    image = read_image(input_path)
    image = stretch_image(image, num_bits)
    write_image(image, output_path)


if __name__ == "__main__":
    app()
