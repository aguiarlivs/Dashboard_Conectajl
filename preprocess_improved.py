# preprocess_improved.py
# Gera data_trilhas.json a partir de:
#   - progresso.xlsx
#   - pontos.xlsx
#   - produtos_provas.csv
#
# Regras principais:
# - "manual" identificado exclusivamente pela coluna TIPO (case-insensitive)
# - Progresso da trilha no mês = média das médias de progresso por curso
#   considerando o snapshot ACUMULADO ATÉ O MÊS (≤ mês)
# - Sem rateio de pontos entre cursos
# - Linha "Pontos Manuais" apenas no Top cursos — visão geral (overall)
# - Valores de pontos sempre inteiros (arredondados)
# - Cursos COM/SEM prova definidos *somente* via produtos_provas.csv:
#     • se CSV diz que NÃO possui prova → "Não possui prova"
#     • se possui prova e aluno não fez → "Não realizada"
#     • se possui prova e aluno fez → nota arredondada (texto)
#
# NOTAS (muito importante):
# - KPIs de notas (notaMedia, taxaComProva, taxaAprovacao) são calculados
#   EXCLUSIVAMENTE a partir de:
#       • eventos `prova_aprovada` no pontos.xlsx (por mês)
#       • coluna NOTA do progresso.xlsx (snapshot_global)
# - Progresso e notas agora são 100% desacoplados: mudar uma regra de
#   progresso não altera a nota média, e vice-versa.

import pandas as pd
import numpy as np
import json, re, math
from datetime import datetime, timezone

# -------------------------------------------------------------------
# Configurações
# -------------------------------------------------------------------
NOTA_APROVACAO = 70.0  # corte global de aprovação (pode ajustar)

# -------------------------------------------------------------------
# Helpers genéricos
# -------------------------------------------------------------------
def norm_text(s):
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def curso_norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def norm_key(s):
    return norm_text(s).lower()

def find_col(df, candidates):
    cols = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        got = cols.get(norm_key(cand))
        if got:
            return got
    return None

def ym_from_any(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    try:
        if isinstance(d, (int, float)) and not pd.isna(d):
            dt = pd.to_datetime(d, origin="1899-12-30", unit="D", errors="coerce")
        else:
            dt = pd.to_datetime(d, dayfirst=True, errors="coerce")
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"

def fnum(x, default=0.0):
    try:
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", ".")
        v = float(x)
        return default if np.isnan(v) else v
    except Exception:
        return default

# -------------------------------------------------------------------
# 1) Leitura do CSV de produtos com / sem prova
# -------------------------------------------------------------------
try:
    produtos_provas = pd.read_csv("produtos_provas.csv")
    produtos_provas.columns = [norm_text(c) for c in produtos_provas.columns]

    col_curso = find_col(produtos_provas, ["curso", "nome_produto", "produto", "nome do produto"])
    col_flag  = find_col(produtos_provas, ["possui_prova", "tem_prova", "tem prova", "possui prova"])

    if not col_curso or not col_flag:
        print("⚠️ 'produtos_provas.csv' não possui colunas de curso + possui_prova reconhecíveis.")
        print("   Esperado algo como 'curso' + 'possui_prova'. Todos os cursos serão tratados como SEM prova.")
        PROVA_MAP = {}
    else:
        def parse_flag(v):
            s = str(v).strip().lower()
            if s in ("sim", "s", "yes", "y", "1", "true", "verdadeiro"):
                return True
            if s in ("nao", "não", "n", "no", "0", "false", "falso", ""):
                return False
            return False

        tmp = produtos_provas[[col_curso, col_flag]].copy()
        tmp["curso_norm"] = tmp[col_curso].astype(str).apply(curso_norm)
        tmp["has_prova"]  = tmp[col_flag].apply(parse_flag)
        tmp = tmp[tmp["curso_norm"] != ""]
        PROVA_MAP = (
            tmp.drop_duplicates("curso_norm", keep="last")
               .set_index("curso_norm")["has_prova"]
               .to_dict()
        )
        print(f"ℹ️ {len(PROVA_MAP)} cursos mapeados (com/sem prova).")

except FileNotFoundError:
    print("⚠️ Aviso: arquivo 'produtos_provas.csv' não encontrado. "
          "Todos os cursos serão tratados como SEM prova.")
    PROVA_MAP = {}

def curso_tem_prova(nome_curso: str) -> bool:
    """Retorna True se o curso TEM prova segundo o CSV (caso contrário, False)."""
    if not nome_curso:
        return False
    k = curso_norm(nome_curso)
    return PROVA_MAP.get(k, False)

# -------------------------------------------------------------------
# 2) Ler planilhas de PROGRESSO e PONTOS
# -------------------------------------------------------------------
progresso = pd.read_excel("progresso.xlsx")
pontos    = pd.read_excel("pontos.xlsx")

progresso.columns = [norm_text(c) for c in progresso.columns]
pontos.columns    = [norm_text(c) for c in pontos.columns]

# PROGRESSO
P_ALUNO = find_col(progresso, ["aluno","nome do aluno","usuário","usuario","nome"])
P_CURSO = find_col(progresso, ["curso","produto","formação","formacao"])
P_TRILHA= find_col(progresso, ["trilha"])
P_PROG  = find_col(progresso, ["progresso (%)","progresso","progress"])
P_DT    = find_col(progresso, [
    "atualizado em","última atualização","ultima atualizacao","updated at",
    "data","data de atualização","data atualizacao","ultimo acesso","último acesso"
])
P_NOTA = find_col(progresso, [
    "nota da prova", "nota prova", "nota (%)",
    "resultado da prova", "resultado"
])

# PONTOS – formato: CRIADO EM | TAG | ALUNO | TRILHA | VITRINE | CURSO | AULA | TIPO | PONTOS
T_ALUNO = find_col(pontos, ["aluno","nome","usuário","usuario"])
T_TIPO  = find_col(pontos, ["tipo","evento"])
T_PONTOS= find_col(pontos, ["pontos"])
T_DATA  = find_col(pontos, ["criado em","data","created at"])
T_CURSO = find_col(pontos, ["curso","produto","formação","formacao"])
T_TRILHA= find_col(pontos, ["trilha"])

# -------------------------------------------------------------------
# 3) Canonicalizar PROGRESSO e PONTOS
# -------------------------------------------------------------------
def map_progresso(df):
    rows = []
    for _, r in df.iterrows():
        aluno = r.get(P_ALUNO) if P_ALUNO else None
        curso = r.get(P_CURSO) if P_CURSO else None
        trilha = r.get(P_TRILHA) if P_TRILHA else None
        prog = r.get(P_PROG) if P_PROG else None
        dt   = r.get(P_DT)   if P_DT   else None
        nota = r.get(P_NOTA) if P_NOTA else None

        if not aluno or not curso:
            continue

        rows.append({
            "aluno": norm_text(aluno),
            "curso": norm_text(curso),
            "trilha": norm_text(trilha) if trilha and str(trilha).strip() else "(sem trilha)",
            "progresso": fnum(prog, 0.0),
            "nota": fnum(nota, default=np.nan) if nota not in (None, "") else np.nan,
            "atualizado_em": dt,
            "ym": ym_from_any(dt)
        })
    return pd.DataFrame(rows)

def map_pontos(df):
    rows = []
    for _, r in df.iterrows():
        aluno = r.get(T_ALUNO) if T_ALUNO else None
        tipo  = r.get(T_TIPO)  if T_TIPO  else ""
        pts   = r.get(T_PONTOS)if T_PONTOS else 0
        dt    = r.get(T_DATA)  if T_DATA  else None
        curso = r.get(T_CURSO) if T_CURSO else ""
        trilha= r.get(T_TRILHA)if T_TRILHA else None
        if not aluno:
            continue

        tipo_s = norm_text(tipo)
        is_manual = (tipo_s.lower() == "manual")  # detecção via TIPO

        rows.append({
            "aluno": norm_text(aluno),
            "tipo":  tipo_s,
            "pontos": fnum(pts, 0.0),
            "criado_em": dt,
            "curso": norm_text(curso),  # pode ser ""
            "trilha_origem": norm_text(trilha) if trilha and str(trilha).strip() else None,
            "is_manual": is_manual,
            "ym": ym_from_any(dt)
        })
    out = pd.DataFrame(rows)
    return out[out["ym"].notna()] if len(out) else out

P = map_progresso(progresso)
T = map_pontos(pontos)

# Normalização de trilhas
if len(P) and "trilha" in P.columns:
    P["trilha"] = P["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

if len(T) and "trilha" in T.columns:
    T["trilha"] = T["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )
elif len(T) and "trilha_origem" in T.columns:
    T["trilha_origem"] = T["trilha_origem"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

# -------------------------------------------------------------------
# 4) Mapeamento curso → trilha
# -------------------------------------------------------------------
m1 = P[["curso","trilha"]].dropna().drop_duplicates() if "trilha" in P else pd.DataFrame(columns=["curso","trilha"])
m2 = pd.DataFrame(columns=["curso","trilha"])
if len(T):
    m2 = T[T["curso"]!=""][["curso"]].drop_duplicates()
    if len(m2):
        m2["trilha"] = m2["curso"].map(pd.Series(P.set_index("curso")["trilha"]).to_dict())

curso_to_trilha = (
    pd.concat([m1, m2], ignore_index=True)
      .drop_duplicates("curso")
      .set_index("curso")["trilha"]
      .to_dict()
)

if len(P):
    P["trilha"] = P["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))

if len(T):
    mapped = T["curso"].apply(lambda c: curso_to_trilha.get(c, None) if c!="" else None)
    T["trilha"] = np.where(
        T["curso"]!="",
        mapped.fillna("(sem trilha)"),
        T["trilha_origem"].fillna("(sem trilha)")
    )

# -------------------------------------------------------------------
# 5) Meses e snapshots de progresso (para PROGRESSO apenas)
# -------------------------------------------------------------------
meses = sorted(set(P["ym"].dropna().unique()).union(set(T["ym"].dropna().unique())))
if not meses:
    raise SystemExit("Sem meses válidos.")

P_sorted = P.copy()
P_sorted["_dt"] = pd.to_datetime(P_sorted["atualizado_em"], errors="coerce")

# Snapshot global (último registro conhecido por aluno-curso)
snapshot_global = (
    P_sorted.sort_values(["aluno","curso","_dt"])
            .groupby(["aluno","curso"], as_index=False).tail(1)
)
snapshot_global["trilha"] = snapshot_global["curso"].map(
    lambda c: curso_to_trilha.get(c, "(sem trilha)")
)

# Último progresso NO MÊS (ym == mês) – usado apenas para saber se houve update
last_in_month = {}
for m in meses:
    dfm = P_sorted[P_sorted["ym"] == m].sort_values(["aluno","curso","_dt"])
    if len(dfm):
        last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
        last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
        last_in_month[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()
    else:
        last_in_month[m] = pd.DataFrame(columns=["aluno","curso","trilha","progresso","nota"])

# Snapshot acumulado por mês (≤ mês) – base oficial para progresso
last_by_month_snapshot = {}
for m in meses:
    dfm = P_sorted[(P_sorted["ym"].isna()) | (P_sorted["ym"] <= m)].sort_values(["aluno","curso","_dt"])
    last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
    last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
    last_by_month_snapshot[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()

# -------------------------------------------------------------------
# 6) Conjunto de cursos com prova (via CSV)
# -------------------------------------------------------------------
if "curso" in P.columns:
    cursos_com_prova = {
        c for c in P["curso"].dropna().unique()
        if curso_tem_prova(c)
    }
else:
    cursos_com_prova = set()

# -------------------------------------------------------------------
# 7) Eventos de prova_aprovada + nota real (NOTAS apenas)
# -------------------------------------------------------------------
provas_evt = T[
    (T["curso"] != "") &
    (T["tipo"].str.contains("prova_aprovada", case=False, na=False))
].copy()

if len(provas_evt):
    provas_evt["_dt"] = pd.to_datetime(provas_evt["criado_em"], errors="coerce")
    provas_evt["trilha"] = provas_evt["curso"].map(
        lambda c: curso_to_trilha.get(c, "(sem trilha)")
    )

    # Uma linha por (aluno, curso, ym) – último evento do mês
    provas_evt = (
        provas_evt.sort_values(["aluno", "curso", "ym", "_dt"])
                  .groupby(["aluno", "curso", "ym"], as_index=False)
                  .tail(1)
    )

    # Anexa NOTA do snapshot_global (nota final do curso)
    snap_notas = snapshot_global[["aluno", "curso", "nota"]].copy()
    provas_evt = provas_evt.merge(
        snap_notas, on=["aluno", "curso"], how="left", suffixes=("", "_snap")
    )
    provas_evt["nota"] = provas_evt["nota"].apply(lambda v: fnum(v, default=np.nan))
else:
    provas_evt = pd.DataFrame(columns=["ym", "aluno", "curso", "trilha", "nota"])

# -------------------------------------------------------------------
# 8) Séries por trilha (PROGRESSO + NOTA desacoplados)
# -------------------------------------------------------------------
trilhas = sorted(
    set((P["trilha"] if "trilha" in P else pd.Series(dtype=str)).dropna().map(norm_text).unique())
    .union(
    set((T["trilha"] if "trilha" in T else pd.Series(dtype=str)).dropna().map(norm_text).unique()))
) or ["(sem trilha)"]

series_trilhas = {}
for tr in trilhas:
    progMed, hadUpdate, pontos_m, concl_m, ativos_m = [], [], [], [], []
    nota_med_tr, taxa_com_prova_tr, taxa_aprov_tr = [], [], []

    for m in meses:
        # ---------- PROGRESSO (snapshot acumulado ≤ mês) ----------
        snap = last_by_month_snapshot[m]
        snap_tr = snap[snap["trilha"] == tr]

        if len(snap_tr):
            by_course = snap_tr.groupby("curso", as_index=False)["progresso"].mean()
            val = float(by_course["progresso"].mean())
        else:
            val = 0.0
        progMed.append(round(val, 2))

        # houve atualização naquele mês?
        lm = last_in_month[m]
        had = bool(len(lm[lm["trilha"] == tr]))
        hadUpdate.append(had)

        # pontos e conclusões (no mês)
        tm = T[(T["ym"] == m) & (T["trilha"] == tr)]
        pontos_m.append(int(np.rint(tm["pontos"].sum())) if len(tm) else 0)
        concl_m.append(int(tm["tipo"].str.contains("curso_concluido", case=False, na=False).sum()) if len(tm) else 0)

        # alunos ativos no mês
        prog_mes = P_sorted[(P_sorted["ym"] == m) & (P_sorted["trilha"] == tr)]
        pts_mes  = tm
        ativos_alunos = set(prog_mes["aluno"].dropna().unique()).union(
            set(pts_mes["aluno"].dropna().unique())
        )
        ativos_m.append(len(ativos_alunos))

        # ---------- NOTAS (somente eventos de prova daquele mês) ----------
        provas_tr_m = provas_evt[
            (provas_evt["ym"] == m) &
            (provas_evt["trilha"] == tr) &
            (provas_evt["nota"].notna())
        ]
        notas_vals = provas_tr_m["nota"].dropna().tolist()

        if notas_vals:
            media = float(np.mean(notas_vals))
            qtd_provas = len(notas_vals)
            aprovados = sum(v >= NOTA_APROVACAO for v in notas_vals)

            taxa_cp = 1.0            # houve prova → 100%
            taxa_aprov = aprovados / max(qtd_provas, 1)
        else:
            media = 0.0
            taxa_cp = 0.0
            taxa_aprov = 0.0

        nota_med_tr.append(round(media, 2))
        taxa_com_prova_tr.append(round(taxa_cp * 100.0, 1))
        taxa_aprov_tr.append(round(taxa_aprov * 100.0, 1))

    # Δ mês a mês para progresso
    progDelta = []
    for i, _ in enumerate(meses):
        if i == 0:
            progDelta.append(0.0)
            continue
        delta = round(progMed[i] - progMed[i-1], 2)
        progDelta.append(delta if hadUpdate[i] else 0.0)

    series_trilhas[tr] = {
        "meses": meses,
        "progMed": progMed,
        "progDelta": progDelta,
        "progHadUpdate": hadUpdate,
        "pontos": pontos_m,
        "conclusoes": concl_m,
        "ativos": ativos_m,
        "notaMedia": nota_med_tr,
        "taxaComProva": taxa_com_prova_tr,
        "taxaAprovacao": taxa_aprov_tr
    }

# -------------------------------------------------------------------
# 9) Top cursos por trilha + visão geral (usa snapshot_global)
# -------------------------------------------------------------------
lib_trilha = P.groupby(["trilha","curso"], as_index=False)["aluno"].nunique()\
              .rename(columns={"aluno":"alunos_liberados"})

prog_trilha = snapshot_global.groupby(["trilha","curso"], as_index=False)\
                             .agg(progresso_medio=("progresso","mean"))
prog_trilha["progresso_medio"] = prog_trilha["progresso_medio"].round(2)

if "nota" in snapshot_global.columns:
    snap_provas = snapshot_global[
        snapshot_global["curso"].isin(cursos_com_prova) &
        snapshot_global["nota"].notna()
    ].copy()
    if len(snap_provas):
        g_nt = snap_provas.groupby(["trilha","curso"], as_index=False)
        nota_trilha = g_nt.agg(
            nota_media=("nota", lambda x: float(np.nanmean(x)) if x.notna().any() else np.nan),
            provas_realizadas=("nota", lambda x: int(x.notna().sum())),
            aprovados=("nota", lambda x: int((x >= NOTA_APROVACAO).sum()))
        )
        nota_trilha["taxa_aprovacao"] = nota_trilha.apply(
            lambda r: (r["aprovados"]/r["provas_realizadas"]*100.0) if r["provas_realizadas"]>0 else 0.0,
            axis=1
        )
        nota_trilha["nota_media"] = nota_trilha["nota_media"].round(2)
    else:
        nota_trilha = pd.DataFrame(columns=["trilha","curso","nota_media",
                                            "provas_realizadas","aprovados","taxa_aprovacao"])
else:
    nota_trilha = pd.DataFrame(columns=["trilha","curso","nota_media",
                                        "provas_realizadas","aprovados","taxa_aprovacao"])

pts_trilha  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
               .groupby(["trilha","curso"], as_index=False)["pontos"].sum()
               .rename(columns={"pontos":"pontos_total"}))

top_cursos_global = {}
for tr in trilhas:
    base = (lib_trilha[lib_trilha["trilha"]==tr]
              .merge(pts_trilha[pts_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .merge(prog_trilha[prog_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .merge(nota_trilha[nota_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .fillna({
                  "pontos_total":0,
                  "progresso_medio":0,
                  "nota_media":0,
                  "provas_realizadas":0,
                  "aprovados":0,
                  "taxa_aprovacao":0
              }))
    base["pontos_total"] = np.rint(base["pontos_total"]).astype(int)
    base = base.sort_values(
        ["progresso_medio","alunos_liberados","pontos_total"],
        ascending=[False, False, False]
    )

    rows = base[[
        "curso",
        "alunos_liberados",
        "progresso_medio",
        "pontos_total",
        "nota_media",
        "provas_realizadas",
        "taxa_aprovacao"
    ]].to_dict(orient="records")
    top_cursos_global[tr] = rows

# visão geral (sem trilha)
lib_all = P.groupby(["curso"], as_index=False)["aluno"].nunique().rename(columns={"aluno":"alunos_liberados"})
prog_all = snapshot_global.groupby(["curso"], as_index=False).agg(progresso_medio=("progresso","mean"))
prog_all["progresso_medio"] = prog_all["progresso_medio"].round(2)

if "nota" in snapshot_global.columns:
    snap_provas_all = snapshot_global[
        snapshot_global["curso"].isin(cursos_com_prova) &
        snapshot_global["nota"].notna()
    ].copy()
    if len(snap_provas_all):
        g_all = snap_provas_all.groupby(["curso"], as_index=False)
        nota_all = g_all.agg(
            nota_media=("nota", lambda x: float(np.nanmean(x)) if x.notna().any() else np.nan),
            provas_realizadas=("nota", lambda x: int(x.notna().sum())),
            aprovados=("nota", lambda x: int((x >= NOTA_APROVACAO).sum()))
        )
        nota_all["taxa_aprovacao"] = nota_all.apply(
            lambda r: (r["aprovados"]/r["provas_realizadas"]*100.0) if r["provas_realizadas"]>0 else 0.0,
            axis=1
        )
        nota_all["nota_media"] = nota_all["nota_media"].round(2)
    else:
        nota_all = pd.DataFrame(columns=["curso","nota_media","provas_realizadas","aprovados","taxa_aprovacao"])
else:
    nota_all = pd.DataFrame(columns=["curso","nota_media","provas_realizadas","aprovados","taxa_aprovacao"])

pts_all  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
            .groupby(["curso"], as_index=False)["pontos"].sum()
            .rename(columns={"pontos":"pontos_total"}))

overall_base = (lib_all
    .merge(pts_all, on="curso", how="left")
    .merge(prog_all, on="curso", how="left")
    .merge(nota_all, on="curso", how="left")
    .fillna({
        "pontos_total":0,
        "progresso_medio":0,
        "nota_media":0,
        "provas_realizadas":0,
        "aprovados":0,
        "taxa_aprovacao":0
    }))
overall_base["pontos_total"] = np.rint(overall_base["pontos_total"]).astype(int)
overall_base = overall_base.sort_values(
    ["progresso_medio","alunos_liberados","pontos_total"],
    ascending=[False,False,False]
)

manual_pts_total = int(np.rint(T[T["is_manual"] == True]["pontos"].sum())) if len(T) else 0
manual_users = int(T[T["is_manual"] == True]["aluno"].nunique()) if len(T) else 0
overall_top_cursos = overall_base[[
    "curso",
    "alunos_liberados",
    "progresso_medio",
    "pontos_total",
    "nota_media",
    "provas_realizadas",
    "taxa_aprovacao"
]].to_dict(orient="records")

if manual_pts_total > 0 or manual_users > 0:
    overall_top_cursos = ([{
        "curso": "Pontos Manuais",
        "alunos_liberados": manual_users,
        "progresso_medio": "Não se aplica",
        "pontos_total": manual_pts_total,
        "nota_media": "Não se aplica",
        "provas_realizadas": "Não se aplica",
        "taxa_aprovacao": "Não se aplica"
    }] + overall_top_cursos)

# -------------------------------------------------------------------
# 10) Tabela de alunos por mês (pontos + nota_prova textual)
# -------------------------------------------------------------------
pts_a_c_t_mes = (
    T[(T["is_manual"] == False) & (T["curso"] != "")]
      .groupby(["ym", "aluno", "curso", "trilha"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_mes"})
)

if len(pts_a_c_t_mes):
    pts_a_c_t_mes["ym"] = pd.Categorical(pts_a_c_t_mes["ym"], categories=meses, ordered=True)
    pts_a_c_t_mes = pts_a_c_t_mes.sort_values(["aluno","curso","trilha","ym"])
    pts_a_c_t_mes["pontos_cum"] = pts_a_c_t_mes.groupby(["aluno","curso","trilha"])["pontos_mes"].cumsum()
else:
    pts_a_c_t_mes["pontos_cum"] = pd.Series(dtype=float)

manuals_a_mes = (
    T[T["is_manual"] == True]
      .groupby(["ym", "aluno"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_manua_mes"})
)

if len(manuals_a_mes):
    manuals_a_mes["ym"] = pd.Categorical(manuals_a_mes["ym"], categories=meses, ordered=True)
    manuals_a_mes = manuals_a_mes.sort_values(["aluno","ym"])
    manuals_a_mes["pontos_manua_cum"] = manuals_a_mes.groupby(["aluno"])["pontos_manua_mes"].cumsum()
else:
    manuals_a_mes["pontos_manua_cum"] = pd.Series(dtype=float)

students = {}
manuals   = {}

for m in meses:
    snap = last_by_month_snapshot[m].copy()

    if len(pts_a_c_t_mes):
        pts_cum_le_m = pts_a_c_t_mes[pts_a_c_t_mes["ym"] <= m]
        if len(pts_cum_le_m):
            pts_cum_last = (
                pts_cum_le_m
                .sort_values(["aluno","curso","trilha","ym"])
                .groupby(["aluno","curso","trilha"], as_index=False)
                .tail(1)[["aluno","curso","trilha","pontos_cum"]]
            )
        else:
            pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])
    else:
        pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])

    base = snap.merge(pts_cum_last, on=["aluno","curso","trilha"], how="outer")

    if "progresso" not in base.columns:
        base["progresso"] = 0.0
    base["progresso"] = base["progresso"].fillna(0.0)

    if "pontos_cum" not in base.columns:
        base["pontos_cum"] = 0
    base["pontos_cum"] = base["pontos_cum"].fillna(0)

    if "nota" not in base.columns:
        base["nota"] = np.nan

    def resolve_nota(row):
        curso = row.get("curso") or ""
        nota  = row.get("nota")

        if not curso_tem_prova(curso):
            return "Não possui prova"

        if nota is None or (isinstance(nota, float) and math.isnan(nota)):
            return "Não realizada"

        try:
            v = fnum(nota)
            if math.isnan(v):
                return "Não realizada"
            return str(int(round(v)))
        except Exception:
            return "Não realizada"

    base["nota_prova"] = base.apply(resolve_nota, axis=1)

    base = base.rename(columns={"pontos_cum": "pontos_mes"})
    out_cols = ["aluno", "curso", "trilha", "progresso", "nota_prova", "pontos_mes"]
    base = base[out_cols].sort_values(["trilha", "aluno", "curso"], kind="stable")
    students[m] = base.to_dict(orient="records")

    if len(manuals_a_mes):
        man_le_m = manuals_a_mes[manuals_a_mes["ym"] <= m]
        if len(man_le_m):
            man_last = (
                man_le_m
                .sort_values(["aluno","ym"])
                .groupby(["aluno"], as_index=False)
                .tail(1)[["aluno","pontos_manua_cum"]]
            )
            man_last["pontos_manua_cum"] = np.rint(man_last["pontos_manua_cum"]).astype(int)
        else:
            man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])
    else:
        man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])

    if len(man_last):
        man_last = man_last.rename(columns={"pontos_manua_cum": "pontos_manua_mes"})
    manuals[m] = man_last.sort_values(["aluno"], kind="stable").to_dict(orient="records")

# -------------------------------------------------------------------
# 11) Séries gerais (overall) – PROGRESSO + NOTAS desacoplados
# -------------------------------------------------------------------
prog_o, pontos_o, concl_o, ativos_o = [], [], [], []
nota_media_o, taxa_com_prova_o, taxa_aprov_o = [], [], []

for m in meses:
    # PROGRESSO geral do mês (snapshot acumulado ≤ mês)
    last_snap = last_by_month_snapshot[m]
    vals = last_snap["progresso"].dropna().tolist()
    prog_o.append(round(float(np.mean(vals)) if vals else 0.0, 2))

    tm = T[T["ym"] == m]
    pontos_o.append(int(np.rint(tm["pontos"].sum())))
    concl_o.append(int(tm["tipo"].str.contains("curso_concluido", case=False, na=False).sum()))

    prog_mes = P_sorted[P_sorted["ym"] == m]
    ativos_alunos = set(prog_mes["aluno"].dropna().unique()).union(
        set(tm["aluno"].dropna().unique())
    )
    ativos_o.append(len(ativos_alunos))

    # NOTAS gerais do mês (somente provas_aprovadas + nota)
    provas_m = provas_evt[
        (provas_evt["ym"] == m) &
        (provas_evt["nota"].notna())
    ]
    notas_m = provas_m["nota"].dropna().tolist()
    if notas_m:
        media = float(np.mean(notas_m))
        qtd_provas = len(notas_m)
        aprovados = sum(v >= NOTA_APROVACAO for v in notas_m)

        taxa_cp = 1.0   # houve prova → 100%
        taxa_aprov = aprovados / max(qtd_provas, 1)
    else:
        media = 0.0
        taxa_cp = 0.0
        taxa_aprov = 0.0

    nota_media_o.append(round(media, 2))
    taxa_com_prova_o.append(round(taxa_cp * 100.0, 1))
    taxa_aprov_o.append(round(taxa_aprov * 100.0, 1))

# -------------------------------------------------------------------
# 12) Montar saída final
# -------------------------------------------------------------------
out = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "meses": meses,
    "overall": {
        "meses": meses,
        "progMed": prog_o,
        "pontos": pontos_o,
        "conclusoes": concl_o,
        "ativos": ativos_o,
        "notaMedia": nota_media_o,
        "taxaComProva": taxa_com_prova_o,
        "taxaAprovacao": taxa_aprov_o
    },
    "overall_top_cursos": overall_top_cursos,
    "trilhas": {
        tr: {
            "series": series_trilhas[tr],
            "top_cursos_global": top_cursos_global.get(tr, [])
        } for tr in trilhas
    },
    "students": students,
    "manuals": manuals
}

with open("data_trilhas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("✅ data_trilhas.json gerado com sucesso — PROGRESSO e NOTAS calculados com regras independentes.")
