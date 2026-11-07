"""Jalankan salinan code-fp.ipynb, lalu simpan data di balik grafik paper ke figdata.npz.

Notebook aslinya tidak diubah. Butuh waktu sekitar 6-7 menit di CPU.

    python capture_figdata.py [--out FOLDER]
"""
import argparse
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell

HERE = Path(__file__).resolve().parent

DUMP_CELL = """
_m = best_models['TCN']
_layer = next(l.name for l in _m.layers if 'conv1d' in l.name.lower())
_maps = tf.keras.Model(inputs=_m.input, outputs=_m.get_layer(_layer).output).predict(X_test[-1:], verbose=0)
_arrays = {
    'y': np.asarray(y_test_inv),
    'shap_usd': np.asarray(shap_values_usd),
    'x_aggr': np.asarray(X_test_aggr),
    'features': np.asarray(feature_names),
    'labels': np.asarray(labels),
    'x_last': np.asarray(X_test[-1, :, 0]),
    'maps': np.asarray(_maps[0]),
}
for _name in best_predictions:
    _arrays[f'pred_{_name}'] = np.asarray(inverse_close(best_predictions[_name]))
    _arrays[f'tsne_{_name}'] = np.asarray(_tsne[_name])
    _arrays[f'best_rmse_{_name}'] = np.asarray(results_df.loc[results_df['Model'] == _name, 'Best RMSE'].iloc[0])
np.savez_compressed(_FIGDATA_PATH, **_arrays)
print('figdata tersimpan')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE, help="folder tujuan figdata.npz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(HERE.parent / "code-fp.ipynb", as_version=4)
    patched = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        if "tsne_results = TSNE(" in cell.source:
            # Hasil t-SNE di notebook hanya variabel lokal per model, jadi disimpan di sini
            cell.source = "_tsne = {}\n" + cell.source.replace(
                ".fit_transform(latent_features)\n",
                ".fit_transform(latent_features)\n    _tsne[name] = tsne_results\n",
            )
            patched += 1
    assert patched == 1, "sel t-SNE tidak ditemukan"

    target = (args.out / "figdata.npz").as_posix()
    nb.cells.append(new_code_cell(f"_FIGDATA_PATH = {target!r}\n" + DUMP_CELL))

    start = time.time()
    with tempfile.TemporaryDirectory() as workdir:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": workdir}}).execute()
    print(f"Selesai dalam {(time.time() - start) / 60:.1f} menit: {target}")


if __name__ == "__main__":
    main()
