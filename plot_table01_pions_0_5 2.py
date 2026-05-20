from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("10.17182/HEPData-ins1759506-v1-csv/")
TABLE_FILE = DATA_DIR / "Table01.csv"  # centralidade 0.0-5.0 para píons em Pb-Pb


def read_hepdata_csv(path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("#:"):
            body = line[2:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip()] = value.strip()
        elif line.startswith("#"):
            continue
        elif line.strip():
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Nenhuma tabela encontrada em {path}")

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[data_start:])))
    df.columns = [c.strip() for c in df.columns]
    return metadata, df


def main() -> None:
    metadata, df = read_hepdata_csv(TABLE_FILE)

    # Colunas físicas da tabela 1
    x = pd.to_numeric(df[r"$p_{T}$ [$GeV/c$]"], errors="coerce")
    xlow = pd.to_numeric(df[r"$p_{T}$ [$GeV/c$] LOW"], errors="coerce")
    xhigh = pd.to_numeric(df[r"$p_{T}$ [$GeV/c$] HIGH"], errors="coerce")
    y = pd.to_numeric(df[r"(1/Nev)*D2(N)/DPT/DYRAP [$(GeV/c)^{-1}$]"], errors="coerce")

    stat = pd.to_numeric(df["stat. +"], errors="coerce").abs()
    syst = pd.to_numeric(df["syst. +"], errors="coerce").abs()
    syst_uncorr = pd.to_numeric(df["syst. uncorr. +"], errors="coerce").abs()

    # Parte correlacionada inferida da decomposição quadrática
    syst_corr = np.sqrt(np.clip(syst**2 - syst_uncorr**2, a_min=0.0, a_max=None))

    mask = x.notna() & xlow.notna() & xhigh.notna() & y.notna()
    x = x[mask].to_numpy()
    xlow = xlow[mask].to_numpy()
    xhigh = xhigh[mask].to_numpy()
    y = y[mask].to_numpy()
    stat = stat[mask].to_numpy()
    syst = syst[mask].to_numpy()
    syst_uncorr = syst_uncorr[mask].to_numpy()
    syst_corr = syst_corr[mask].to_numpy()

    xerr = np.vstack([x - xlow, xhigh - x])

    # Figura com 2 painéis:
    # 1) espectro com bandas de erro
    # 2) erros relativos (%) para comparar os três tipos
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.5], "hspace": 0.06}
    )

    # Painel superior: yield
    ax.fill_between(
        x, y - syst_uncorr, y + syst_uncorr,
        alpha=0.28, label="syst. uncorr."
    )
    ax.fill_between(
        x, y - syst_corr, y + syst_corr,
        alpha=0.22, label="syst. corr."
    )
    ax.errorbar(
        x, y, xerr=xerr, yerr=stat,
        fmt="o", ms=3.2, capsize=2, label="stat."
    )
    ax.plot(x, y, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N_{\rm ev})\, d^2N/(dp_T\,dy)$ [$(\mathrm{GeV}/c)^{-1}$]")
    ax.set_title(
        "ALICE Table 01 — Pb–Pb 5.02 TeV, centralidade 0–5%\n"
        "Píons: yield com erros estatístico, sistemático não correlacionado e correlacionado"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Painel inferior: erros relativos
    rel_stat = 100.0 * stat / y
    rel_unc = 100.0 * syst_uncorr / y
    rel_corr = 100.0 * syst_corr / y

    axr.plot(x, rel_stat, "o-", ms=3, label="stat.")
    axr.plot(x, rel_unc, "o-", ms=3, label="syst. uncorr.")
    axr.plot(x, rel_corr, "o-", ms=3, label="syst. corr.")
    axr.set_xscale("log")
    axr.set_xlabel(r"$p_T$ [GeV/$c$]")
    axr.set_ylabel("erro [%]")
    axr.grid(True, alpha=0.3)
    axr.legend()

    out = Path("plot_table01_pions_0_5.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()

    print("Arquivo salvo em:", out.resolve())
    print("Descrição:", metadata.get("description", ""))
    print("DOI:", metadata.get("table_doi", ""))


if __name__ == "__main__":
    main()
