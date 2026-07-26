from irag.tools.plot_legend import plot_legend


def test_plot_legend_writes_horizontal_image(tmp_path):
    output_path = tmp_path / "legend.png"

    result = plot_legend(output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
