from PIL import Image

from irag.tools.compose_plot_grid import compose_plot_grid


def test_compose_plot_grid_supports_partial_final_row(tmp_path):
    plot_paths = []
    colors = ["red", "green", "blue", "yellow", "purple"]
    for index, color in enumerate(colors):
        path = tmp_path / f"plot-{index}.png"
        Image.new("RGB", (10, 8), color).save(path)
        plot_paths.append(path)
    legend_path = tmp_path / "legend.png"
    Image.new("RGBA", (6, 2), (0, 0, 0, 255)).save(legend_path)
    output_path = tmp_path / "grid.png"

    compose_plot_grid(
        plot_paths,
        legend_path,
        output_path,
        columns=3,
    )

    output = Image.open(output_path).convert("RGB")
    assert output.size == (30, 42)
    assert output.getpixel((0, 8)) == (255, 255, 255)
    assert output.getpixel((5, 8)) == (255, 255, 0)
    assert output.getpixel((15, 8)) == (128, 0, 128)
