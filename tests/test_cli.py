from types import SimpleNamespace

import geoscript_ir.__main__ as cli


def test_main_writes_tikz_document(tmp_path, monkeypatch):
    program_path = tmp_path / "scene.gs"
    program_path.write_text("A;", encoding="utf-8")

    model = SimpleNamespace(
        points={"A": (0.0, 0.0)},
        gauges=[],
        residuals=[SimpleNamespace(key="res", size=1, kind="eq")],
    )
    solution = SimpleNamespace(
        success=True,
        max_residual=0.0,
        warnings=[],
        point_coords={"A": (0.0, 0.0)},
    )

    monkeypatch.setattr(cli, "parse_program", lambda text: "program")
    monkeypatch.setattr(cli, "validate", lambda program: None)
    monkeypatch.setattr(cli, "desugar_variants", lambda program: ["variant"])
    monkeypatch.setattr(cli, "print_program", lambda program: "variant IR")
    monkeypatch.setattr(cli, "check_consistency", lambda program: [])
    monkeypatch.setattr(cli, "translate", lambda program: model)
    monkeypatch.setattr(cli, "solve", lambda model, opts: solution)
    monkeypatch.setattr(cli, "normalize_point_coords", lambda coords: coords)
    monkeypatch.setattr(cli, "score_solution", lambda solution: 0)

    tikz_path = tmp_path / "out" / "diagram.tex"
    rendered_documents = []

    def _generate_document(program, point_coords, **kwargs):
        rendered_documents.append((program, point_coords, kwargs))
        return "tikz document"

    monkeypatch.setattr(cli, "generate_tikz_document", _generate_document)

    cli.main([str(program_path), "--tikz-output-path", str(tikz_path)])

    assert tikz_path.read_text(encoding="utf-8") == "tikz document"
    assert rendered_documents == [("variant", {"A": (0.0, 0.0)}, {"normalize": True})]
