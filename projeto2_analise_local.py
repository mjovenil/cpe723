# projeto2_analise.py
# Script de análise e visualização dos resultados — DA vs ES vs Híbrido
#
# Lê os CSVs e arquivos .npy salvos pelas simulações e gera gráficos
# com qualidade de publicação.
#
# Dependências:
#   pip install numpy matplotlib pandas seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# =========================================================
# 0. Configuração visual global
# =========================================================
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

# Paleta consistente para os três algoritmos
COR_DA  = "#1f77b4"   # azul
COR_ES  = "#d62728"   # vermelho
COR_HIB = "#2ca02c"   # verde
COR_REF = "#7f7f7f"   # cinza (referência)

# =========================================================
# 1. Caminhos — ajuste conforme sua máquina
# =========================================================
local_path_da  = os.path.expanduser("~/cpe723_da")
local_path_es  = os.path.expanduser("~/cpe723_es")
local_path_hib = os.path.expanduser("~/cpe723_hibrido")
output_path    = os.path.expanduser("~/cpe723_analise")
os.makedirs(output_path, exist_ok=True)

csv_da  = os.path.join(local_path_da,  "grid_search_da_results.csv")
csv_es  = os.path.join(local_path_es,  "grid_search_es_results.csv")
csv_hib = os.path.join(local_path_hib, "grid_search_hibrido_results.csv")

J_ref = 0.030216   # custo com centros verdadeiros (NC=8, seed=1)

# =========================================================
# 2. Carrega dados
# =========================================================
def load_csv(path, label):
    if not os.path.exists(path):
        print(f"[AVISO] CSV não encontrado: {path}")
        return None
    df = pd.read_csv(path)
    print(f"{label}: {len(df)} combinações carregadas")
    return df

df_da  = load_csv(csv_da,  "DA")
df_es  = load_csv(csv_es,  "ES")
df_hib = load_csv(csv_hib, "Híbrido")

def load_npy(path, key="best_hist_Dbest"):
    """Carrega arquivo .npy da melhor combinação."""
    if pd.isna(path) or not os.path.exists(str(path)):
        return None
    return np.load(str(path))

# =========================================================
# 3. Gráfico 1 — Comparativo SR, MBF, Tempo (barras)
#    Melhor combinação de cada algoritmo
# =========================================================
def grafico_comparativo_barras(df_da, df_es, df_hib):

    melhor_da  = df_da.sort_values("SR", ascending=False).iloc[0]
    melhor_es  = df_es.sort_values("SR", ascending=False).iloc[0]

    algoritmos = ["DA", "ES"]
    cores      = [COR_DA, COR_ES]
    sr_vals    = [melhor_da["SR"]*100, melhor_es["SR"]*100]
    mbf_vals   = [melhor_da["MBF"],    melhor_es["MBF"]]
    tempo_vals = [melhor_da["tempo_medio_s"], melhor_es["tempo_medio_s"]]

    if df_hib is not None and len(df_hib) > 0:
        melhor_hib = df_hib.sort_values("SR", ascending=False).iloc[0]
        if melhor_hib["SR"] > 0:
            algoritmos.append("Híbrido")
            cores.append(COR_HIB)
            sr_vals.append(melhor_hib["SR"]*100)
            mbf_vals.append(melhor_hib["MBF"])
            tempo_vals.append(melhor_hib["tempo_medio_s"])

    x = np.arange(len(algoritmos))
    w = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Comparativo de Desempenho — Melhor Configuração por Algoritmo ($NC=8$)",
                 fontsize=13, fontweight="bold", y=1.02)

    # SR
    bars = axes[0].bar(x, sr_vals, width=w, color=cores, edgecolor="white", linewidth=0.8)
    axes[0].axhline(100, color=COR_REF, linestyle="--", linewidth=0.8, label="Máximo")
    axes[0].set_xticks(x); axes[0].set_xticklabels(algoritmos)
    axes[0].set_ylabel("Taxa de Sucesso (%)"); axes[0].set_title("SR")
    axes[0].set_ylim(0, 110)
    for bar, val in zip(bars, sr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # MBF
    bars = axes[1].bar(x, mbf_vals, width=w, color=cores, edgecolor="white", linewidth=0.8)
    axes[1].axhline(J_ref, color=COR_REF, linestyle="--", linewidth=0.8, label=f"$J_{{ref}}={J_ref:.4f}$")
    axes[1].set_xticks(x); axes[1].set_xticklabels(algoritmos)
    axes[1].set_ylabel("MBF"); axes[1].set_title("MBF (menor = melhor)")
    axes[1].legend(fontsize=9)
    for bar, val in zip(bars, mbf_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Tempo
    bars = axes[2].bar(x, tempo_vals, width=w, color=cores, edgecolor="white", linewidth=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(algoritmos)
    axes[2].set_ylabel("Tempo médio (s)"); axes[2].set_title("Tempo de Execução")
    axes[2].set_yscale("log")
    for bar, val in zip(bars, tempo_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                     f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_path, "comparativo_barras.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 4. Gráfico 2 — Boxplot SR × Algoritmo
# =========================================================
def grafico_boxplot_sr(df_da, df_es, df_hib):
    """
    Boxplot da distribuição de SR em todas as combinações
    testadas de cada algoritmo.
    """
    dados = []
    for df, label in [(df_da, "DA"), (df_es, "ES")]:
        if df is not None:
            for sr in df["SR"].dropna():
                dados.append({"Algoritmo": label, "SR (%)": sr * 100})
    if df_hib is not None:
        for sr in df_hib["SR"].dropna():
            dados.append({"Algoritmo": "Híbrido", "SR (%)": sr * 100})

    df_plot = pd.DataFrame(dados)
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"DA": COR_DA, "ES": COR_ES, "Híbrido": COR_HIB}
    sns.boxplot(data=df_plot, x="Algoritmo", y="SR (%)",
                palette=palette, width=0.5, linewidth=1.2,
                flierprops=dict(marker="o", markersize=4, alpha=0.5), ax=ax)
    sns.stripplot(data=df_plot, x="Algoritmo", y="SR (%)",
                  palette=palette, size=4, alpha=0.4, jitter=True, ax=ax)
    ax.set_title("Distribuição de SR — Todas as Combinações Testadas ($NC=8$)",
                 fontweight="bold")
    ax.set_ylabel("Taxa de Sucesso (%)")
    ax.set_xlabel("")

    plt.tight_layout()
    out = os.path.join(output_path, "boxplot_sr.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 5. Gráfico 3 — Scatter SR vs Tempo (trade-off)
# =========================================================
def grafico_scatter_sr_tempo(df_da, df_es, df_hib):
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, label, cor in [(df_da, "DA", COR_DA),
                            (df_es, "ES", COR_ES),
                            (df_hib, "Híbrido", COR_HIB)]:
        if df is None or len(df) == 0:
            continue
        df_v = df.dropna(subset=["SR", "tempo_medio_s"])
        ax.scatter(df_v["tempo_medio_s"], df_v["SR"] * 100,
                   color=cor, label=label, s=60, alpha=0.8,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Tempo médio por execução (s)")
    ax.set_ylabel("Taxa de Sucesso (%)")
    ax.set_title("Trade-off SR × Tempo de Execução ($NC=8$)", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(output_path, "scatter_sr_tempo.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 6. Gráfico 4 — Heatmap SR do ES (Nind × Nfilhos)
# =========================================================
def grafico_heatmap_es(df_es):
    if df_es is None:
        return

    df_v = df_es.dropna(subset=["SR", "Nind", "Nfilhos"])
    if df_v.empty:
        return

    pivot = df_v.pivot_table(
        values="SR", index="Nind", columns="Nfilhos",
        aggfunc="max"
    ) * 100

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "SR (%)"},
                ax=ax)
    ax.set_title("Heatmap SR (%) — ES: $N_{ind}$ × $N_{filhos}$ ($NC=8$)",
                 fontweight="bold")
    ax.set_xlabel("$N_{filhos}$")
    ax.set_ylabel("$N_{ind}$")

    plt.tight_layout()
    out = os.path.join(output_path, "heatmap_sr_es.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 7. Gráfico 5 — Heatmap SR do Híbrido (T_da × alpha_da)
# =========================================================
def grafico_heatmap_hibrido(df_hib):
    if df_hib is None or len(df_hib) < 2:
        print("Híbrido: dados insuficientes para heatmap (rode o grid completo).")
        return

    pivot = df_hib.pivot_table(
        values="SR", index="T_da", columns="alpha_da",
        aggfunc="max"
    ) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "SR (%)"},
                ax=ax)
    ax.set_title(r"Heatmap SR (%) — Híbrido: $T_{DA}$ × $\alpha_{DA}$ ($NC=8$)",
                 fontweight="bold")
    ax.set_xlabel(r"$\alpha_{DA}$ (intensidade do guiamento)")
    ax.set_ylabel(r"$T_{DA}$ (temperatura do passo DA)")

    plt.tight_layout()
    out = os.path.join(output_path, "heatmap_sr_hibrido.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 8. Gráfico 6 — Curvas de convergência comparativas
#    DA vs ES (melhor de cada) — eixo x em tempo (s)
# =========================================================
def grafico_convergencia_comparativo(df_da, df_es):
    if df_da is None or df_es is None:
        return

    melhor_da = df_da.sort_values("SR", ascending=False).iloc[0]
    melhor_es = df_es.sort_values("SR", ascending=False).iloc[0]

    hist_da_dbest = load_npy(melhor_da.get("history_Dbest_file"))
    hist_da_time  = load_npy(melhor_da.get("history_time_file"))
    hist_es_dbest = load_npy(melhor_es.get("history_Dbest_file"))
    hist_es_time  = load_npy(melhor_es.get("history_time_file"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Convergência da Melhor Configuração — DA vs ES ($NC=8$)",
                 fontweight="bold", fontsize=13)

    # Painel esquerdo: eixo x = iterações/gerações
    ax = axes[0]
    if hist_da_dbest is not None:
        ax.plot(hist_da_dbest, color=COR_DA, linewidth=1.8, label="DA")
    if hist_es_dbest is not None:
        ax.plot(hist_es_dbest, color=COR_ES, linewidth=1.8, label="ES")
    ax.axhline(J_ref, color=COR_REF, linestyle="--", linewidth=1.0,
               label=f"$J_{{ref}}={J_ref:.4f}$")
    ax.set_xlabel("Iteração / Geração")
    ax.set_ylabel("Melhor custo acumulado ($D_{{best}}$)")
    ax.set_title("Por iteração/geração")
    ax.legend()

    # Painel direito: eixo x = tempo (s)
    ax = axes[1]
    if hist_da_dbest is not None and hist_da_time is not None:
        ax.plot(hist_da_time, hist_da_dbest, color=COR_DA, linewidth=1.8, label="DA")
    if hist_es_dbest is not None and hist_es_time is not None:
        ax.plot(hist_es_time, hist_es_dbest, color=COR_ES, linewidth=1.8, label="ES")
    ax.axhline(J_ref, color=COR_REF, linestyle="--", linewidth=1.0,
               label=f"$J_{{ref}}={J_ref:.4f}$")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Melhor custo acumulado ($D_{{best}}$)")
    ax.set_title("Por tempo de execução")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(output_path, "convergencia_comparativa.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 9. Gráfico 7 — Tabela visual de resultados finais
# =========================================================
def grafico_tabela_resultados(df_da, df_es, df_hib):
    melhor_da = df_da.sort_values("SR", ascending=False).iloc[0]
    melhor_es = df_es.sort_values("SR", ascending=False).iloc[0]

    linhas = [
        ["DA",
         f"$T_0={melhor_da['T0']},\\;\\alpha={melhor_da['alpha']}$",
         f"{melhor_da['SR']*100:.0f}\\%",
         f"{melhor_da['MBF']:.4f}",
         f"{melhor_da['AES']:.0f}",
         f"{melhor_da['tempo_medio_s']*1000:.1f} ms"],
        ["ES",
         f"$N_{{ind}}={melhor_es['Nind']},\\;N_{{fil}}={melhor_es['Nfilhos']}$",
         f"{melhor_es['SR']*100:.0f}\\%",
         f"{melhor_es['MBF']:.4f}",
         f"{melhor_es['AES']:.0f}",
         f"{melhor_es['tempo_medio_s']:.2f} s"],
    ]

    if df_hib is not None and len(df_hib) > 0:
        melhor_hib = df_hib.sort_values("SR", ascending=False).iloc[0]
        if melhor_hib["SR"] > 0:
            linhas.append([
                "Híbrido",
                f"$T_{{DA}}={melhor_hib['T_da']},\\;\\alpha_{{DA}}={melhor_hib['alpha_da']}$",
                f"{melhor_hib['SR']*100:.0f}\\%",
                f"{melhor_hib['MBF']:.4f}",
                f"{melhor_hib['AES']:.0f}" if not np.isnan(melhor_hib['AES']) else "—",
                f"{melhor_hib['tempo_medio_s']:.2f} s",
            ])

    cols = ["Algoritmo", "Parâmetros", "SR", "MBF", "AES", "Tempo"]

    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.6 * len(linhas)))
    ax.axis("off")

    tbl = ax.table(
        cellText=linhas,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Estilo do cabeçalho
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Linhas alternadas
    cores_linha = [COR_DA, COR_ES, COR_HIB]
    for i, linha in enumerate(linhas, start=1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(cores_linha[i-1] + "22")

    ax.set_title("Tabela Comparativa — Melhores Configurações ($NC=8$)",
                 fontweight="bold", fontsize=13, pad=10)

    plt.tight_layout()
    out = os.path.join(output_path, "tabela_resultados.png")
    plt.savefig(out); plt.show()
    print(f"Salvo: {out}")

# =========================================================
# 10. Execução
# =========================================================
if __name__ == "__main__":

    print(f"\nGráficos serão salvos em: {output_path}\n")
    print("=" * 50)

    print("\n[1/7] Comparativo de barras (SR, MBF, Tempo)...")
    grafico_comparativo_barras(df_da, df_es, df_hib)

    print("\n[2/7] Boxplot SR por algoritmo...")
    grafico_boxplot_sr(df_da, df_es, df_hib)

    print("\n[3/7] Scatter SR vs Tempo...")
    grafico_scatter_sr_tempo(df_da, df_es, df_hib)

    print("\n[4/7] Heatmap SR — ES (Nind x Nfilhos)...")
    grafico_heatmap_es(df_es)

    print("\n[5/7] Heatmap SR — Híbrido (T_da x alpha_da)...")
    grafico_heatmap_hibrido(df_hib)

    print("\n[6/7] Curvas de convergência comparativas...")
    grafico_convergencia_comparativo(df_da, df_es)

    print("\n[7/7] Tabela visual de resultados...")
    grafico_tabela_resultados(df_da, df_es, df_hib)

    print(f"\nConcluído. Todos os gráficos salvos em: {output_path}")