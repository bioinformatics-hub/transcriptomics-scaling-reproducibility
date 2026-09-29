"""Public entry point must work without private scratch modules or data."""
import main


def test_array_preprocessing_uses_selected_config(tmp_path, monkeypatch):
    folder = tmp_path / 'configs'
    folder.mkdir()
    for name in ['config_0.yml', 'config_1.yml']:
        (folder / name).write_text('{}')
    calls = []
    monkeypatch.setattr(main, 'compute_hvgs', lambda path, **kw: calls.append(('hvg', path)))
    monkeypatch.setattr(main, 'compute_means', lambda path, **kw: calls.append(('means', path)))
    monkeypatch.setattr(main, 'train_from_dir', lambda *a, **kw: calls.append(('train', a)))
    main.run_train(['--config-dir', str(folder), '--id', '1', '-A'])
    assert calls[:2] == [('hvg', str(folder / 'config_1.yml')), ('means', str(folder / 'config_1.yml'))]
    assert calls[2][0] == 'train'
